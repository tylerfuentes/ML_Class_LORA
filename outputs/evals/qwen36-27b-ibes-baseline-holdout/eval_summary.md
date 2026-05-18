# Eval Summary

## wrds_holdout

### base / thinking

- accuracy: `0.0100`
- macro_f1: `0.0202`
- parse_failure_rate: `0.9900`
- average_output_tokens: `80.00`
- contains_think_rate: `0.0000`
- malformed_think_tag_rate: `0.0000`
- parseable_after_strip_rate: `0.0100`

### adapter / thinking

- accuracy: `0.0000`
- macro_f1: `0.0000`
- parse_failure_rate: `1.0000`
- average_output_tokens: `80.00`
- contains_think_rate: `0.0000`
- malformed_think_tag_rate: `0.0000`
- parseable_after_strip_rate: `0.0000`

### base / nonthinking

- accuracy: `0.4200`
- macro_f1: `0.5183`
- parse_failure_rate: `0.5600`
- average_output_tokens: `80.00`
- contains_think_rate: `0.0000`
- malformed_think_tag_rate: `0.0000`
- parseable_after_strip_rate: `0.4400`

### adapter / nonthinking

- accuracy: `1.0000`
- macro_f1: `1.0000`
- parse_failure_rate: `0.0000`
- average_output_tokens: `28.00`
- contains_think_rate: `0.0000`
- malformed_think_tag_rate: `0.0000`
- parseable_after_strip_rate: `1.0000`

### Qwen Thinking Mode Comparison

- `nonthinking` base accuracy=`0.4200` adapter accuracy=`1.0000` delta=`+0.5800`
- `nonthinking` base macro_f1=`0.5183` adapter macro_f1=`1.0000` delta=`+0.4817`
- `nonthinking` parse failure delta=`-0.5600` think-tag delta=`+0.0000`
- `nonthinking` readiness verdict: `improved`
- `thinking` base accuracy=`0.0100` adapter accuracy=`0.0000` delta=`-0.0100`
- `thinking` base macro_f1=`0.0202` adapter macro_f1=`0.0000` delta=`-0.0202`
- `thinking` parse failure delta=`+0.0100` think-tag delta=`+0.0000`
- `thinking` readiness verdict: `regressed`
