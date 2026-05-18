# Eval Summary

## wrds_holdout

### base / nonthinking

- accuracy: `0.4490`
- macro_f1: `0.5405`
- parse_failure_rate: `0.5340`
- average_output_tokens: `79.99`
- truncated_output_rate: `0.9990`
- contains_think_rate: `0.0000`
- malformed_think_tag_rate: `0.0000`
- json_after_reasoning_text_rate: `0.0000`
- parseable_after_strip_rate: `0.4660`
- used_chat_template: `True`
- chat_template_effective: `True`

### adapter / nonthinking

- accuracy: `1.0000`
- macro_f1: `1.0000`
- parse_failure_rate: `0.0000`
- average_output_tokens: `28.00`
- truncated_output_rate: `0.0000`
- contains_think_rate: `0.0000`
- malformed_think_tag_rate: `0.0000`
- json_after_reasoning_text_rate: `0.0000`
- parseable_after_strip_rate: `1.0000`
- used_chat_template: `True`
- chat_template_effective: `True`

### Qwen Thinking Mode Comparison

- `nonthinking` base accuracy=`0.4490` adapter accuracy=`1.0000` delta=`+0.5510`
- `nonthinking` base macro_f1=`0.5405` adapter macro_f1=`1.0000` delta=`+0.4595`
- `nonthinking` parse failure delta=`-0.5340` think-tag delta=`+0.0000`
- `nonthinking` readiness verdict: `improved`
