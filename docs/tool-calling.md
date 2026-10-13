# Tool calling and templates

Keep the model's original chat_template.jinja, tokenizer files and both EOS IDs
from generation_config.json unchanged. The recipe uses TabbyAPI tool_format=auto
and the packaged template, not a Flash Next template copied from another recipe.

SixCat's published Tools row is 90/100 over 20 items. That is evidence for the
publisher's actual server, not a guarantee for every parser or client.

Standard OpenAI tools requests require a compatible server parser. Test at least:
a normal answer with zero tool calls, one declared function, numeric/boolean
argument types, a tool-result round trip, and streamed assembly. Treat returned
arguments as untrusted until validated. Do not execute real commands just to
check parsing. Required/named/parallel tool-choice behavior is not newly verified
by this recipe.

Thinking in the SixCat run was enabled at temperature 0.7. The speed benchmark
was greedy. They are not the same sampling configuration. For smoke requests,
chat_template_kwargs.enable_thinking=false is sent explicitly; validate the
actual rendered prompt and reasoning/content split in the adopted server.
