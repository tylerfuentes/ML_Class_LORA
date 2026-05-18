# Eval Summary

## benchmark_tfns

### base / nonthinking

- accuracy: `0.5234`
- macro_f1: `0.5581`
- parse_failure_rate: `0.0156`
- average_output_tokens: `64.99`
- truncated_output_rate: `0.1406`
- contains_think_rate: `0.0000`
- malformed_think_tag_rate: `0.0000`
- json_after_reasoning_text_rate: `0.0000`
- parseable_after_strip_rate: `0.9844`
- used_chat_template: `True`
- chat_template_effective: `True`

### adapter / nonthinking

- accuracy: `0.6953`
- macro_f1: `0.6970`
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

- `nonthinking` base accuracy=`0.5234` adapter accuracy=`0.6953` delta=`+0.1719`
- `nonthinking` base macro_f1=`0.5581` adapter macro_f1=`0.6970` delta=`+0.1389`
- `nonthinking` parse failure delta=`-0.0156` think-tag delta=`+0.0000`
- `nonthinking` readiness verdict: `improved`
