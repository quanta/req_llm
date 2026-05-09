defmodule ReqLLM.Providers.Anthropic.Response do
  @moduledoc """
  Anthropic-specific response decoding for the Messages API format.

  Handles decoding Anthropic Messages API responses to ReqLLM structures.

  ## Anthropic Response Format

      %{
        "id" => "msg_01XFDUDYJgAACzvnptvVoYEL",
        "type" => "message",
        "role" => "assistant",
        "model" => "claude-sonnet-4-5-20250929",
        "content" => [
          %{"type" => "text", "text" => "Hello! How can I help you today?"}
        ],
        "stop_reason" => "stop",
        "stop_sequence" => nil,
        "usage" => %{
          "input_tokens" => 10,
          "output_tokens" => 20
        }
      }

  ## Streaming Format

  Anthropic uses Server-Sent Events (SSE) with different event types:
  - message_start: Initial message metadata
  - content_block_start: Start of content block
  - content_block_delta: Incremental content
  - content_block_stop: End of content block
  - message_delta: Final message updates
  - message_stop: End of message

  """

  @doc """
  Decode Anthropic response data to ReqLLM.Response.
  """
  @spec decode_response(map(), LLMDB.Model.t()) :: {:ok, ReqLLM.Response.t()} | {:error, term()}
  def decode_response(data, model) when is_map(data) do
    id = Map.get(data, "id", "unknown")
    model_name = Map.get(data, "model", model.id || "unknown")
    usage = parse_usage(Map.get(data, "usage"))

    finish_reason = parse_finish_reason(Map.get(data, "stop_reason"))

    content_chunks = decode_content(Map.get(data, "content", []))
    message = build_message_from_chunks(content_chunks)

    context = %ReqLLM.Context{
      messages: if(message, do: [message], else: [])
    }

    response = %ReqLLM.Response{
      id: id,
      model: model_name,
      context: context,
      message: message,
      stream?: false,
      stream: nil,
      usage: usage,
      finish_reason: finish_reason,
      provider_meta: Map.drop(data, ["id", "model", "content", "usage", "stop_reason"])
    }

    {:ok, response}
  end

  def decode_response(_data, _model) do
    {:error, :not_implemented}
  end

  @doc """
  Decode Anthropic SSE event data into StreamChunks.
  """
  @spec decode_stream_event(map(), LLMDB.Model.t()) :: [ReqLLM.StreamChunk.t()]
  def decode_stream_event(%{data: data}, _model) when is_map(data) do
    case data do
      %{"type" => "message_start", "message" => message} ->
        usage_data = Map.get(message, "usage", %{})

        if usage_data == %{} do
          []
        else
          usage = parse_usage(usage_data)
          [ReqLLM.StreamChunk.meta(%{usage: usage})]
        end

      %{"type" => "content_block_delta", "index" => index, "delta" => delta} ->
        decode_content_block_delta(delta, index)

      %{"type" => "content_block_start", "index" => index, "content_block" => block} ->
        decode_content_block_start(block, index)

      # Terminal events with metadata
      %{"type" => "message_stop"} ->
        [ReqLLM.StreamChunk.meta(%{terminal?: true})]

      %{"type" => "message_delta", "delta" => delta} ->
        finish_reason =
          case Map.get(delta, "stop_reason") do
            "end_turn" -> :stop
            "max_tokens" -> :length
            "stop_sequence" -> :stop
            "tool_use" -> :tool_calls
            "pause_turn" -> :pause_turn
            _ -> :unknown
          end

        usage = Map.get(data, "usage", %{})

        chunks = [ReqLLM.StreamChunk.meta(%{finish_reason: finish_reason, terminal?: true})]

        # Add usage chunk if present
        if usage == %{} do
          chunks
        else
          usage_chunk = ReqLLM.StreamChunk.meta(%{usage: usage})
          [usage_chunk | chunks]
        end

      %{"type" => "ping"} ->
        # Keep-alive ping, no content
        []

      _ ->
        []
    end
  end

  def decode_stream_event(_, _model), do: []

  # Private helper functions

  defp decode_content([]), do: []

  defp decode_content(content) when is_list(content) do
    content
    |> Enum.map(&decode_content_block/1)
    |> List.flatten()
    |> Enum.reject(&is_nil/1)
  end

  defp decode_content(content) when is_binary(content) do
    [ReqLLM.StreamChunk.text(content)]
  end

  defp decode_content_block(%{"type" => "text", "text" => text}) do
    ReqLLM.StreamChunk.text(text)
  end

  defp decode_content_block(%{"type" => "thinking", "thinking" => text} = block) do
    ReqLLM.StreamChunk.thinking(text, thinking_metadata(block))
  end

  defp decode_content_block(%{"type" => "thinking", "text" => text} = block) do
    ReqLLM.StreamChunk.thinking(text, thinking_metadata(block))
  end

  defp decode_content_block(
         %{"type" => "tool_use", "id" => id, "name" => name, "input" => input} = block
       ) do
    meta = put_caller(%{id: id}, block)
    ReqLLM.StreamChunk.tool_call(name, input, meta)
  end

  defp decode_content_block(%{"type" => "compaction", "content" => content}) do
    %ReqLLM.StreamChunk{type: :compaction, text: content}
  end

  defp decode_content_block(%{"type" => "server_tool_use"} = block) do
    %ReqLLM.StreamChunk{type: :content, text: "", metadata: %{raw_block: block}}
  end

  defp decode_content_block(%{"type" => "tool_search_tool_result"} = block) do
    %ReqLLM.StreamChunk{type: :content, text: "", metadata: %{raw_block: block}}
  end

  defp decode_content_block(%{"type" => "bash_code_execution_tool_result"} = block) do
    %ReqLLM.StreamChunk{type: :content, text: "", metadata: %{raw_block: block}}
  end

  defp decode_content_block(%{"type" => "code_execution_tool_result"} = block) do
    %ReqLLM.StreamChunk{type: :content, text: "", metadata: %{raw_block: block}}
  end

  defp decode_content_block(%{"type" => "text_editor_code_execution"} = block) do
    %ReqLLM.StreamChunk{type: :content, text: "", metadata: %{raw_block: block}}
  end

  defp decode_content_block(%{"type" => "text_editor_code_execution_tool_result"} = block) do
    %ReqLLM.StreamChunk{type: :content, text: "", metadata: %{raw_block: block}}
  end

  defp decode_content_block(%{"type" => "advisor_tool_result"} = block) do
    %ReqLLM.StreamChunk{type: :content, text: "", metadata: %{raw_block: block}}
  end

  defp decode_content_block(_), do: nil

  defp decode_content_block_delta(%{"type" => "text_delta", "text" => text}, _index)
       when is_binary(text) do
    [ReqLLM.StreamChunk.text(text)]
  end

  defp decode_content_block_delta(%{"type" => "thinking_delta", "thinking" => text}, _index)
       when is_binary(text) do
    [ReqLLM.StreamChunk.thinking(text, thinking_metadata())]
  end

  defp decode_content_block_delta(%{"type" => "thinking_delta", "text" => text}, _index)
       when is_binary(text) do
    [ReqLLM.StreamChunk.thinking(text, thinking_metadata())]
  end

  defp decode_content_block_delta(
         %{"type" => "input_json_delta", "partial_json" => fragment},
         index
       )
       when is_binary(fragment) do
    # Accumulate JSON fragments; StreamResponse.extract_tool_calls will merge these
    [ReqLLM.StreamChunk.meta(%{tool_call_args: %{index: index, fragment: fragment}})]
  end

  defp decode_content_block_delta(%{"type" => "compaction_delta", "content" => content}, _index)
       when is_binary(content) do
    [%ReqLLM.StreamChunk{type: :compaction, text: content}]
  end

  defp decode_content_block_delta(_, _index), do: []

  defp decode_content_block_start(%{"type" => "text", "text" => text}, _index) do
    [ReqLLM.StreamChunk.text(text)]
  end

  defp decode_content_block_start(%{"type" => "thinking", "thinking" => text}, _index) do
    [ReqLLM.StreamChunk.thinking(text, thinking_metadata())]
  end

  defp decode_content_block_start(%{"type" => "thinking", "text" => text}, _index) do
    [ReqLLM.StreamChunk.thinking(text, thinking_metadata())]
  end

  defp decode_content_block_start(
         %{"type" => "tool_use", "id" => id, "name" => name} = block,
         index
       ) do
    # Tool call start - send empty arguments that will be filled by deltas
    meta = put_caller(%{id: id, index: index, start: true}, block)
    [ReqLLM.StreamChunk.tool_call(name, %{}, meta)]
  end

  defp decode_content_block_start(%{"type" => "compaction"}, _index) do
    [%ReqLLM.StreamChunk{type: :compaction, text: ""}]
  end

  defp decode_content_block_start(%{"type" => "server_tool_use"} = block, _index) do
    [%ReqLLM.StreamChunk{type: :content, text: "", metadata: %{raw_block: block}}]
  end

  defp decode_content_block_start(%{"type" => "tool_search_tool_result"} = block, _index) do
    [%ReqLLM.StreamChunk{type: :content, text: "", metadata: %{raw_block: block}}]
  end

  defp decode_content_block_start(%{"type" => "bash_code_execution_tool_result"} = block, _index) do
    [%ReqLLM.StreamChunk{type: :content, text: "", metadata: %{raw_block: block}}]
  end

  defp decode_content_block_start(%{"type" => "code_execution_tool_result"} = block, _index) do
    [%ReqLLM.StreamChunk{type: :content, text: "", metadata: %{raw_block: block}}]
  end

  defp decode_content_block_start(%{"type" => "text_editor_code_execution"} = block, _index) do
    [%ReqLLM.StreamChunk{type: :content, text: "", metadata: %{raw_block: block}}]
  end

  defp decode_content_block_start(
         %{"type" => "text_editor_code_execution_tool_result"} = block,
         _index
       ) do
    [%ReqLLM.StreamChunk{type: :content, text: "", metadata: %{raw_block: block}}]
  end

  defp decode_content_block_start(%{"type" => "advisor_tool_result"} = block, _index) do
    [%ReqLLM.StreamChunk{type: :content, text: "", metadata: %{raw_block: block}}]
  end

  defp decode_content_block_start(_, _index), do: []

  defp build_message_from_chunks([]), do: nil

  defp build_message_from_chunks(chunks) do
    content_parts =
      chunks
      |> Enum.filter(&(&1.type in [:content, :thinking, :compaction]))
      |> Enum.map(&chunk_to_content_part/1)
      |> Enum.reject(&is_nil/1)

    tool_calls =
      chunks
      |> Enum.filter(&(&1.type == :tool_call))
      |> Enum.map(&chunk_to_tool_call/1)
      |> Enum.reject(&is_nil/1)

    reasoning_details = extract_reasoning_details(chunks)

    if content_parts != [] or tool_calls != [] do
      %ReqLLM.Message{
        role: :assistant,
        content: content_parts,
        tool_calls: if(tool_calls != [], do: tool_calls),
        reasoning_details: if(reasoning_details != [], do: reasoning_details),
        metadata: %{}
      }
    end
  end

  defp extract_reasoning_details(chunks) do
    chunks
    |> Enum.filter(&(&1.type == :thinking))
    |> Enum.with_index()
    |> Enum.map(fn {chunk, index} ->
      sig = Map.get(chunk.metadata, :signature)

      %ReqLLM.Message.ReasoningDetails{
        text: chunk.text,
        signature: sig,
        encrypted?: sig != nil,
        provider: :anthropic,
        format: "anthropic-thinking-v1",
        index: index,
        provider_data: %{"type" => "thinking"}
      }
    end)
  end

  defp chunk_to_content_part(%ReqLLM.StreamChunk{
         type: :content,
         text: _text,
         metadata: %{raw_block: _block} = metadata
       }) do
    %ReqLLM.Message.ContentPart{type: :text, text: "", metadata: metadata}
  end

  defp chunk_to_content_part(%ReqLLM.StreamChunk{type: :content, text: text}) do
    %ReqLLM.Message.ContentPart{type: :text, text: text}
  end

  defp chunk_to_content_part(%ReqLLM.StreamChunk{type: :thinking, text: text}) do
    %ReqLLM.Message.ContentPart{type: :thinking, text: text}
  end

  defp chunk_to_content_part(%ReqLLM.StreamChunk{type: :compaction, text: text}) do
    %ReqLLM.Message.ContentPart{type: :compaction, text: text}
  end

  defp chunk_to_content_part(_), do: nil

  defp chunk_to_tool_call(%ReqLLM.StreamChunk{
         type: :tool_call,
         name: name,
         arguments: args,
         metadata: meta
       }) do
    args_json = if is_binary(args), do: args, else: Jason.encode!(args)
    id = Map.get(meta, :id)
    call = ReqLLM.ToolCall.new(id, name, args_json)

    case Map.get(meta, :caller) do
      nil -> call
      caller -> %{call | function: Map.put(call.function, :caller, caller)}
    end
  end

  defp chunk_to_tool_call(_), do: nil

  # Preserve Anthropic's `caller` field on tool_use blocks (added when a tool is
  # invoked from a code_execution sandbox via `allowed_callers`). Routed through
  # StreamChunk metadata so the encoder can echo it back; without it,
  # server_tool_use blocks become orphaned on round-trip.
  defp put_caller(meta, %{"caller" => caller}), do: Map.put(meta, :caller, caller)
  defp put_caller(meta, _block), do: meta

  defp parse_usage(%{"input_tokens" => input, "output_tokens" => output} = usage) do
    cache_read = Map.get(usage, "cache_read_input_tokens", 0)
    cache_creation = Map.get(usage, "cache_creation_input_tokens", 0)
    reasoning_tokens = Map.get(usage, "reasoning_output_tokens", 0)
    tool_usage = anthropic_tool_usage(usage)
    iterations = parse_iterations(Map.get(usage, "iterations"))

    base = %{
      input_tokens: input,
      output_tokens: output,
      total_tokens: input + output,
      cached_tokens: cache_read,
      cache_read_input_tokens: cache_read,
      cache_creation_input_tokens: cache_creation,
      reasoning_tokens: reasoning_tokens
    }

    base =
      if map_size(tool_usage) > 0 do
        Map.put(base, :tool_usage, tool_usage)
      else
        base
      end

    if iterations == [] do
      base
    else
      Map.put(base, :iterations, iterations)
    end
  end

  defp parse_usage(_),
    do: %{
      input_tokens: 0,
      output_tokens: 0,
      total_tokens: 0,
      cached_tokens: 0,
      reasoning_tokens: 0
    }

  defp anthropic_tool_usage(usage) when is_map(usage) do
    server_tool_use = Map.get(usage, "server_tool_use") || Map.get(usage, :server_tool_use) || %{}

    web_search =
      Map.get(server_tool_use, "web_search_requests") ||
        Map.get(server_tool_use, :web_search_requests)

    tool_search =
      Map.get(server_tool_use, "tool_search_requests") ||
        Map.get(server_tool_use, :tool_search_requests)

    result = %{}

    result =
      if is_number(web_search) and web_search > 0 do
        Map.merge(result, ReqLLM.Usage.Tool.build(:web_search, web_search))
      else
        result
      end

    if is_number(tool_search) and tool_search > 0 do
      Map.put(result, :tool_search_requests, tool_search)
    else
      result
    end
  end

  defp parse_iterations(nil), do: []

  defp parse_iterations(iterations) when is_list(iterations) do
    Enum.map(iterations, &parse_iteration/1)
  end

  defp parse_iterations(_), do: []

  defp parse_iteration(iter) when is_map(iter) do
    %{
      type: Map.get(iter, "type") || Map.get(iter, :type),
      model: Map.get(iter, "model") || Map.get(iter, :model),
      input_tokens: Map.get(iter, "input_tokens") || Map.get(iter, :input_tokens) || 0,
      output_tokens: Map.get(iter, "output_tokens") || Map.get(iter, :output_tokens) || 0,
      cache_read_input_tokens:
        Map.get(iter, "cache_read_input_tokens") || Map.get(iter, :cache_read_input_tokens) || 0,
      cache_creation_input_tokens:
        Map.get(iter, "cache_creation_input_tokens") ||
          Map.get(iter, :cache_creation_input_tokens) || 0
    }
  end

  defp parse_finish_reason("stop"), do: :stop
  defp parse_finish_reason("max_tokens"), do: :length
  defp parse_finish_reason("tool_use"), do: :tool_calls
  defp parse_finish_reason("end_turn"), do: :stop
  defp parse_finish_reason("content_filter"), do: :content_filter
  defp parse_finish_reason("compaction"), do: :compaction
  defp parse_finish_reason("pause_turn"), do: :pause_turn
  defp parse_finish_reason(reason) when is_binary(reason), do: :error
  defp parse_finish_reason(_), do: nil

  defp thinking_metadata(block \\ %{}) do
    signature = Map.get(block, "signature")

    %{
      signature: signature,
      encrypted?: signature != nil,
      provider: :anthropic,
      format: "anthropic-thinking-v1",
      provider_data: %{"type" => "thinking"}
    }
  end
end
