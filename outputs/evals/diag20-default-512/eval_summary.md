# Eval Summary

## wrds_holdout

### base / thinking

- accuracy: `0.7500`
- macro_f1: `0.8410`
- parse_failure_rate: `0.2500`
- average_output_tokens: `512.00`
- truncated_output_rate: `1.0000`
- contains_think_rate: `0.0000`
- malformed_think_tag_rate: `0.0000`
- json_after_reasoning_text_rate: `0.0000`
- parseable_after_strip_rate: `0.7500`
- used_chat_template: `True`
- chat_template_effective: `True`

### adapter / thinking

- accuracy: `0.7000`
- macro_f1: `0.6781`
- parse_failure_rate: `0.0000`
- average_output_tokens: `512.00`
- truncated_output_rate: `1.0000`
- contains_think_rate: `0.0000`
- malformed_think_tag_rate: `0.0000`
- json_after_reasoning_text_rate: `1.0000`
- parseable_after_strip_rate: `1.0000`
- used_chat_template: `True`
- chat_template_effective: `True`

### base / nonthinking

- accuracy: `1.0000`
- macro_f1: `1.0000`
- parse_failure_rate: `0.0000`
- average_output_tokens: `267.20`
- truncated_output_rate: `0.0500`
- contains_think_rate: `0.0000`
- malformed_think_tag_rate: `0.0000`
- json_after_reasoning_text_rate: `0.0000`
- parseable_after_strip_rate: `1.0000`
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

- `nonthinking` base accuracy=`1.0000` adapter accuracy=`1.0000` delta=`+0.0000`
- `nonthinking` base macro_f1=`1.0000` adapter macro_f1=`1.0000` delta=`+0.0000`
- `nonthinking` parse failure delta=`+0.0000` think-tag delta=`+0.0000`
- `nonthinking` readiness verdict: `inconclusive`
- `thinking` base accuracy=`0.7500` adapter accuracy=`0.7000` delta=`-0.0500`
- `thinking` base macro_f1=`0.8410` adapter macro_f1=`0.6781` delta=`-0.1630`
- `thinking` parse failure delta=`-0.2500` think-tag delta=`+0.0000`
- `thinking` readiness verdict: `regressed`
