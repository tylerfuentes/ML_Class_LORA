# Eval Summary

## benchmark_fiqa

### base / nonthinking

- accuracy: `0.7812`
- macro_f1: `0.6491`
- parse_failure_rate: `0.0781`
- average_output_tokens: `64.75`
- truncated_output_rate: `0.1875`
- contains_think_rate: `0.0000`
- malformed_think_tag_rate: `0.0000`
- json_after_reasoning_text_rate: `0.0000`
- parseable_after_strip_rate: `0.9219`
- used_chat_template: `True`
- chat_template_effective: `True`

### adapter / nonthinking

- accuracy: `0.8125`
- macro_f1: `0.6759`
- parse_failure_rate: `0.0000`
- average_output_tokens: `3.22`
- truncated_output_rate: `0.0156`
- contains_think_rate: `0.0000`
- malformed_think_tag_rate: `0.0000`
- json_after_reasoning_text_rate: `0.0000`
- parseable_after_strip_rate: `1.0000`
- used_chat_template: `True`
- chat_template_effective: `True`

### Qwen Thinking Mode Comparison

- `nonthinking` base accuracy=`0.7812` adapter accuracy=`0.8125` delta=`+0.0312`
- `nonthinking` base macro_f1=`0.6491` adapter macro_f1=`0.6759` delta=`+0.0268`
- `nonthinking` parse failure delta=`-0.0781` think-tag delta=`+0.0000`
- `nonthinking` readiness verdict: `improved`
