# Eval Summary

## benchmark_fpb

### base / nonthinking

- accuracy: `0.4688`
- macro_f1: `0.5229`
- parse_failure_rate: `0.0156`
- average_output_tokens: `62.20`
- truncated_output_rate: `0.1094`
- contains_think_rate: `0.0000`
- malformed_think_tag_rate: `0.0000`
- json_after_reasoning_text_rate: `0.0000`
- parseable_after_strip_rate: `0.9844`
- used_chat_template: `True`
- chat_template_effective: `True`

### adapter / nonthinking

- accuracy: `0.7188`
- macro_f1: `0.7553`
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

- `nonthinking` base accuracy=`0.4688` adapter accuracy=`0.7188` delta=`+0.2500`
- `nonthinking` base macro_f1=`0.5229` adapter macro_f1=`0.7553` delta=`+0.2324`
- `nonthinking` parse failure delta=`-0.0156` think-tag delta=`+0.0000`
- `nonthinking` readiness verdict: `improved`
