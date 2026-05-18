# Eval Summary

## wrds_holdout

### base / thinking

- accuracy: `0.4000`
- macro_f1: `0.5480`
- parse_failure_rate: `0.6000`
- average_output_tokens: `256.00`
- truncated_output_rate: `1.0000`
- contains_think_rate: `0.0000`
- malformed_think_tag_rate: `0.0000`
- json_after_reasoning_text_rate: `0.0000`
- parseable_after_strip_rate: `0.4000`
- used_chat_template: `True`
- chat_template_effective: `True`

### adapter / thinking

- accuracy: `0.8000`
- macro_f1: `0.8001`
- parse_failure_rate: `0.0000`
- average_output_tokens: `256.00`
- truncated_output_rate: `1.0000`
- contains_think_rate: `0.0000`
- malformed_think_tag_rate: `0.0000`
- json_after_reasoning_text_rate: `0.1000`
- parseable_after_strip_rate: `1.0000`
- used_chat_template: `True`
- chat_template_effective: `True`

### base / nonthinking

- accuracy: `0.9000`
- macro_f1: `0.9444`
- parse_failure_rate: `0.1000`
- average_output_tokens: `238.55`
- truncated_output_rate: `0.5500`
- contains_think_rate: `0.0000`
- malformed_think_tag_rate: `0.0000`
- json_after_reasoning_text_rate: `0.0000`
- parseable_after_strip_rate: `0.9000`
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

- `nonthinking` base accuracy=`0.9000` adapter accuracy=`1.0000` delta=`+0.1000`
- `nonthinking` base macro_f1=`0.9444` adapter macro_f1=`1.0000` delta=`+0.0556`
- `nonthinking` parse failure delta=`-0.1000` think-tag delta=`+0.0000`
- `nonthinking` readiness verdict: `improved`
- `thinking` base accuracy=`0.4000` adapter accuracy=`0.8000` delta=`+0.4000`
- `thinking` base macro_f1=`0.5480` adapter macro_f1=`0.8001` delta=`+0.2521`
- `thinking` parse failure delta=`-0.6000` think-tag delta=`+0.0000`
- `thinking` readiness verdict: `improved`
