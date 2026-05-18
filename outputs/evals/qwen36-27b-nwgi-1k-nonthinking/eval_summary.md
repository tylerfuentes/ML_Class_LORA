# Eval Summary

## benchmark_nwgi

### base / nonthinking

- accuracy: `0.5234`
- macro_f1: `0.5084`
- parse_failure_rate: `0.0469`
- average_output_tokens: `64.29`
- truncated_output_rate: `0.2031`
- contains_think_rate: `0.0000`
- malformed_think_tag_rate: `0.0000`
- json_after_reasoning_text_rate: `0.0000`
- parseable_after_strip_rate: `0.9531`
- used_chat_template: `True`
- chat_template_effective: `True`

### adapter / nonthinking

- accuracy: `0.6094`
- macro_f1: `0.6023`
- parse_failure_rate: `0.0000`
- average_output_tokens: `2.00`
- truncated_output_rate: `0.0000`
- contains_think_rate: `0.0000`
- malformed_think_tag_rate: `0.0000`
- json_after_reasoning_text_rate: `0.0000`
- parseable_after_strip_rate: `1.0000`
- used_chat_template: `True`
- chat_template_effective: `True`

### Qwen Thinking Mode Comparison

- `nonthinking` base accuracy=`0.5234` adapter accuracy=`0.6094` delta=`+0.0859`
- `nonthinking` base macro_f1=`0.5084` adapter macro_f1=`0.6023` delta=`+0.0938`
- `nonthinking` parse failure delta=`-0.0469` think-tag delta=`+0.0000`
- `nonthinking` readiness verdict: `improved`
