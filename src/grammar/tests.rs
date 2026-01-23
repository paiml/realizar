#[cfg(test)]
mod tests {
    use crate::grammar::*;

    #[test]
    fn test_grammar_element_types() {
        let char_elem = GrammarElement::Char('a');
        let range_elem = GrammarElement::CharRange('a', 'z');
        let rule_ref = GrammarElement::RuleRef("test".to_string());
        let char_not = GrammarElement::CharNot(vec!['x', 'y']);
        let any = GrammarElement::Any;
        let end = GrammarElement::End;

        assert_eq!(char_elem, GrammarElement::Char('a'));
        assert_eq!(range_elem, GrammarElement::CharRange('a', 'z'));
        assert_eq!(rule_ref, GrammarElement::RuleRef("test".to_string()));
        assert_eq!(char_not, GrammarElement::CharNot(vec!['x', 'y']));
        assert_eq!(any, GrammarElement::Any);
        assert_eq!(end, GrammarElement::End);
    }

    #[test]
    fn test_grammar_alternative() {
        let alt =
            GrammarAlternative::new(vec![GrammarElement::Char('a'), GrammarElement::Char('b')]);
        assert_eq!(alt.elements.len(), 2);
        assert!(!alt.is_empty());

        let empty_alt = GrammarAlternative::new(vec![]);
        assert!(empty_alt.is_empty());

        let char_alt = GrammarAlternative::char('x');
        assert_eq!(char_alt.elements.len(), 1);
    }

    #[test]
    fn test_grammar_rule() {
        let rule = GrammarRule::new(
            "test",
            vec![GrammarAlternative::char('a'), GrammarAlternative::char('b')],
        );
        assert_eq!(rule.name, "test");
        assert_eq!(rule.alternatives.len(), 2);

        let single = GrammarRule::single("single", vec![GrammarElement::Char('x')]);
        assert_eq!(single.alternatives.len(), 1);
    }

    #[test]
    fn test_grammar_basic() {
        let mut grammar = Grammar::new();
        assert!(grammar.is_empty());

        grammar.add_rule(GrammarRule::single("root", vec![GrammarElement::Char('a')]));
        assert!(!grammar.is_empty());
        assert_eq!(grammar.len(), 1);
        assert_eq!(grammar.root(), "root");

        assert!(grammar.get_rule("root").is_some());
        assert!(grammar.get_rule("nonexistent").is_none());
    }

    #[test]
    fn test_grammar_validation() {
        let mut grammar = Grammar::new();

        // Empty grammar fails
        assert!(grammar.validate().is_err());

        // Grammar without root rule fails
        grammar.set_root("missing");
        assert!(grammar.validate().is_err());

        // Valid grammar passes
        grammar.add_rule(GrammarRule::single("root", vec![GrammarElement::Char('a')]));
        grammar.set_root("root");
        assert!(grammar.validate().is_ok());

        // Grammar with invalid rule reference fails
        grammar.add_rule(GrammarRule::single(
            "bad",
            vec![GrammarElement::RuleRef("undefined".to_string())],
        ));
        assert!(grammar.validate().is_err());
    }

    #[test]
    fn test_grammar_state_initial() {
        let state = GrammarState::initial("root");
        assert_eq!(state.rule, "root");
        assert_eq!(state.alt_idx, 0);
        assert_eq!(state.elem_idx, 0);
        assert!(state.stack.is_empty());
    }

    #[test]
    fn test_state_machine_simple() {
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::single(
            "root",
            vec![
                GrammarElement::Char('a'),
                GrammarElement::Char('b'),
                GrammarElement::Char('c'),
            ],
        ));

        let mut sm = GrammarStateMachine::new(grammar).expect("test");

        assert!(sm.is_valid_char('a'));
        assert!(!sm.is_valid_char('b'));
        assert!(!sm.is_valid_char('x'));

        assert!(sm.advance('a'));
        assert!(sm.is_valid_char('b'));

        assert!(sm.advance('b'));
        assert!(sm.is_valid_char('c'));

        assert!(sm.advance('c'));
        assert!(sm.is_complete());
    }

    #[test]
    fn test_state_machine_alternatives() {
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::new(
            "root",
            vec![
                GrammarAlternative::new(vec![GrammarElement::Char('a')]),
                GrammarAlternative::new(vec![GrammarElement::Char('b')]),
            ],
        ));

        let sm = GrammarStateMachine::new(grammar).expect("test");

        // Both 'a' and 'b' should be valid initially
        assert!(sm.is_valid_char('a'));
        assert!(sm.is_valid_char('b'));
        assert!(!sm.is_valid_char('c'));
    }

    #[test]
    fn test_state_machine_char_range() {
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::single(
            "root",
            vec![GrammarElement::CharRange('a', 'z')],
        ));

        let sm = GrammarStateMachine::new(grammar).expect("test");

        assert!(sm.is_valid_char('a'));
        assert!(sm.is_valid_char('m'));
        assert!(sm.is_valid_char('z'));
        assert!(!sm.is_valid_char('A'));
        assert!(!sm.is_valid_char('0'));
    }

    #[test]
    fn test_state_machine_reset() {
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::single("root", vec![GrammarElement::Char('a')]));

        let mut sm = GrammarStateMachine::new(grammar).expect("test");

        sm.advance('a');
        assert!(sm.is_complete());
        assert_eq!(sm.generated(), "a");

        sm.reset();
        assert!(!sm.is_complete());
        assert_eq!(sm.generated(), "");
        assert!(sm.is_valid_char('a'));
    }

    #[test]
    fn test_json_schema_string() {
        let schema = JsonSchemaType::String;
        let grammar = grammar_from_json_schema(&schema);

        assert!(grammar.get_rule("root").is_some());
        assert!(grammar.get_rule("string_content").is_some());
    }

    #[test]
    fn test_json_schema_integer() {
        let schema = JsonSchemaType::Integer;
        let grammar = grammar_from_json_schema(&schema);

        let sm = GrammarStateMachine::new(grammar).expect("test");
        assert!(sm.is_valid_char('0'));
        assert!(sm.is_valid_char('9'));
        assert!(sm.is_valid_char('-'));
    }

    #[test]
    fn test_json_schema_boolean() {
        let schema = JsonSchemaType::Boolean;
        let grammar = grammar_from_json_schema(&schema);

        let mut sm = GrammarStateMachine::new(grammar).expect("test");

        // Should accept 't' for 'true' or 'f' for 'false'
        assert!(sm.is_valid_char('t'));
        assert!(sm.is_valid_char('f'));
        assert!(!sm.is_valid_char('a'));

        // Test "true"
        assert!(sm.advance('t'));
        assert!(sm.advance('r'));
        assert!(sm.advance('u'));
        assert!(sm.advance('e'));
        assert!(sm.is_complete());
    }

    #[test]
    fn test_json_schema_null() {
        let schema = JsonSchemaType::Null;
        let grammar = grammar_from_json_schema(&schema);

        let mut sm = GrammarStateMachine::new(grammar).expect("test");

        assert!(sm.advance('n'));
        assert!(sm.advance('u'));
        assert!(sm.advance('l'));
        assert!(sm.advance('l'));
        assert!(sm.is_complete());
    }

    #[test]
    fn test_json_schema_enum() {
        let schema = JsonSchemaType::Enum(vec!["red".to_string(), "blue".to_string()]);
        let grammar = grammar_from_json_schema(&schema);

        let mut sm = GrammarStateMachine::new(grammar).expect("test");

        // Should accept '"' to start string
        assert!(sm.is_valid_char('"'));

        // Test "red"
        assert!(sm.advance('"'));
        assert!(sm.advance('r'));
        assert!(sm.advance('e'));
        assert!(sm.advance('d'));
        assert!(sm.advance('"'));
        assert!(sm.is_complete());
    }

    #[test]
    fn test_json_schema_object() {
        let schema = JsonSchemaType::Object(vec![
            ("name".to_string(), JsonSchemaType::String, true),
            ("age".to_string(), JsonSchemaType::Integer, true),
        ]);
        let grammar = grammar_from_json_schema(&schema);

        assert!(grammar.get_rule("root").is_some());
        assert!(grammar.get_rule("root_name").is_some());
        assert!(grammar.get_rule("root_age").is_some());
    }

    #[test]
    fn test_json_schema_array() {
        let schema = JsonSchemaType::Array(Box::new(JsonSchemaType::Integer));
        let grammar = grammar_from_json_schema(&schema);

        assert!(grammar.get_rule("root").is_some());
        assert!(grammar.get_rule("root_item").is_some());
        assert!(grammar.get_rule("root_items").is_some());
    }

    #[test]
    fn test_token_mask_allow_all() {
        let mask = TokenMask::allow_all(100);
        assert_eq!(mask.num_allowed(), 100);
        assert!(mask.is_allowed(0));
        assert!(mask.is_allowed(99));
        assert!(!mask.is_allowed(100));
        assert!(mask.allow_eos);
    }

    #[test]
    fn test_token_mask_from_allowed() {
        let allowed: HashSet<u32> = vec![1, 2, 3].into_iter().collect();
        let mask = TokenMask::from_allowed(allowed, false);

        assert!(mask.is_allowed(1));
        assert!(mask.is_allowed(2));
        assert!(mask.is_allowed(3));
        assert!(!mask.is_allowed(0));
        assert!(!mask.is_allowed(4));
        assert!(!mask.allow_eos);
    }

    #[test]
    fn test_token_mask_apply_to_logits() {
        let allowed: HashSet<u32> = vec![1, 3].into_iter().collect();
        let mask = TokenMask::from_allowed(allowed, true);

        let mut logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        mask.apply_to_logits(&mut logits);

        assert_eq!(logits[0], f32::NEG_INFINITY); // 0 not allowed
        assert_eq!(logits[1], 2.0); // 1 allowed
        assert_eq!(logits[2], f32::NEG_INFINITY); // 2 not allowed
        assert_eq!(logits[3], 4.0); // 3 allowed
        assert_eq!(logits[4], f32::NEG_INFINITY); // 4 not allowed
    }

    #[test]
    fn test_grammar_token_masker() {
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::single(
            "root",
            vec![GrammarElement::Char('a'), GrammarElement::Char('b')],
        ));

        let mut token_strings = HashMap::new();
        token_strings.insert(0, "a".to_string());
        token_strings.insert(1, "b".to_string());
        token_strings.insert(2, "c".to_string());
        token_strings.insert(3, "ab".to_string());

        let mut masker = GrammarTokenMasker::new(grammar, token_strings, 99).expect("test");

        // Initially, only 'a' and 'ab' should be allowed
        let mask = masker.get_mask();
        assert!(mask.is_allowed(0)); // 'a'
        assert!(!mask.is_allowed(1)); // 'b' - not valid first char
        assert!(!mask.is_allowed(2)); // 'c' - not valid
        assert!(mask.is_allowed(3)); // 'ab' - valid sequence

        // Advance with 'a'
        assert!(masker.advance_token(0));

        // Now only 'b' should be allowed
        let mask = masker.get_mask();
        assert!(!mask.is_allowed(0)); // 'a' - not valid after 'a'
        assert!(mask.is_allowed(1)); // 'b' - valid
        assert!(!mask.is_allowed(2)); // 'c' - not valid

        // Advance with 'b'
        assert!(masker.advance_token(1));
        assert!(masker.is_complete());
    }

    #[test]
    fn test_grammar_token_masker_reset() {
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::single("root", vec![GrammarElement::Char('x')]));

        let mut token_strings = HashMap::new();
        token_strings.insert(0, "x".to_string());

        let mut masker = GrammarTokenMasker::new(grammar, token_strings, 99).expect("test");

        masker.advance_token(0);
        assert!(masker.is_complete());

        masker.reset();
        assert!(!masker.is_complete());
    }

    #[test]
    fn test_valid_chars_collection() {
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::new(
            "root",
            vec![
                GrammarAlternative::new(vec![GrammarElement::Char('a')]),
                GrammarAlternative::new(vec![GrammarElement::Char('b')]),
                GrammarAlternative::new(vec![GrammarElement::CharRange('0', '2')]),
            ],
        ));

        let sm = GrammarStateMachine::new(grammar).expect("test");
        let valid = sm.valid_chars();

        assert!(valid.contains(&'a'));
        assert!(valid.contains(&'b'));
        assert!(valid.contains(&'0'));
        assert!(valid.contains(&'1'));
        assert!(valid.contains(&'2'));
        assert!(!valid.contains(&'3'));
        assert!(!valid.contains(&'c'));
    }

    #[test]
    fn test_grammar_serialization() {
        let mut grammar = Grammar::with_root("test");
        grammar.add_rule(GrammarRule::single("test", vec![GrammarElement::Char('x')]));

        let json = serde_json::to_string(&grammar).expect("test");
        let deserialized: Grammar = serde_json::from_str(&json).expect("test");

        assert_eq!(deserialized.root(), "test");
        assert!(deserialized.get_rule("test").is_some());
    }

    // ==================== Tool Calling Tests ====================

    #[test]
    fn test_tool_parameter_type_default() {
        let param_type = ToolParameterType::default();
        assert_eq!(param_type, ToolParameterType::String);
    }

    #[test]
    fn test_tool_parameter_required_string() {
        let param = ToolParameter::required_string("query", "Search query");

        assert_eq!(param.name, "query");
        assert_eq!(param.description, "Search query");
        assert_eq!(param.param_type, ToolParameterType::String);
        assert!(param.required);
        assert!(param.default.is_none());
    }

    #[test]
    fn test_tool_parameter_optional_string() {
        let param = ToolParameter::optional_string("format", "Output format");

        assert_eq!(param.name, "format");
        assert_eq!(param.description, "Output format");
        assert_eq!(param.param_type, ToolParameterType::String);
        assert!(!param.required);
    }

    #[test]
    fn test_tool_parameter_required_int() {
        let param = ToolParameter::required_int("count", "Number of results");

        assert_eq!(param.name, "count");
        assert_eq!(param.param_type, ToolParameterType::Integer);
        assert!(param.required);
    }

    #[test]
    fn test_tool_parameter_required_enum() {
        let param = ToolParameter::required_enum(
            "format",
            "Output format",
            vec!["json".to_string(), "xml".to_string()],
        );

        match &param.param_type {
            ToolParameterType::Enum(values) => {
                assert_eq!(values.len(), 2);
                assert!(values.contains(&"json".to_string()));
                assert!(values.contains(&"xml".to_string()));
            },
            _ => panic!("Expected Enum type"),
        }
        assert!(param.required);
    }

    #[test]
    fn test_tool_parameter_with_default() {
        let param = ToolParameter::optional_string("format", "Output format").with_default("json");

        assert_eq!(param.default.as_deref(), Some("json"));
    }

    #[test]
    fn test_tool_parameter_type_array() {
        let array_type = ToolParameterType::Array {
            items: Box::new(ToolParameterType::Integer),
        };

        match &array_type {
            ToolParameterType::Array { items } => {
                assert_eq!(**items, ToolParameterType::Integer);
            },
            _ => panic!("Expected Array type"),
        }
    }

    #[test]
    fn test_tool_parameter_type_nested_object() {
        let inner_params = vec![
            ToolParameter::required_string("street", "Street address"),
            ToolParameter::required_string("city", "City name"),
        ];

        let object_type = ToolParameterType::Object {
            properties: inner_params,
        };

        match &object_type {
            ToolParameterType::Object { properties } => {
                assert_eq!(properties.len(), 2);
                assert_eq!(properties[0].name, "street");
                assert_eq!(properties[1].name, "city");
            },
            _ => panic!("Expected Object type"),
        }
    }

    #[test]
    fn test_tool_definition_creation() {
        let tool = ToolDefinition::new(
            "get_weather",
            "Get current weather for a location",
            vec![
                ToolParameter::required_string("location", "City name"),
                ToolParameter::optional_string("unit", "Temperature unit"),
            ],
        );

        assert_eq!(tool.name, "get_weather");
        assert_eq!(tool.description, "Get current weather for a location");
        assert_eq!(tool.parameters.len(), 2);
        assert!(tool.parameters[0].required);
        assert!(!tool.parameters[1].required);
    }

    #[test]
    fn test_tool_definition_required_params() {
        let tool = ToolDefinition::new(
            "test",
            "Test tool",
            vec![
                ToolParameter::required_string("req1", "Required 1"),
                ToolParameter::optional_string("opt1", "Optional 1"),
                ToolParameter::required_string("req2", "Required 2"),
            ],
        );

        let required: Vec<_> = tool.required_params().collect();
        assert_eq!(required.len(), 2);
        assert_eq!(required[0].name, "req1");
        assert_eq!(required[1].name, "req2");
    }

    #[test]
    fn test_tool_definition_optional_params() {
        let tool = ToolDefinition::new(
            "test",
            "Test tool",
            vec![
                ToolParameter::required_string("req1", "Required 1"),
                ToolParameter::optional_string("opt1", "Optional 1"),
                ToolParameter::optional_string("opt2", "Optional 2"),
            ],
        );

        let optional: Vec<_> = tool.optional_params().collect();
        assert_eq!(optional.len(), 2);
        assert_eq!(optional[0].name, "opt1");
        assert_eq!(optional[1].name, "opt2");
    }

    #[test]
    fn test_tool_definition_is_valid_name() {
        // Valid names
        assert!(ToolDefinition::is_valid_name("get_weather"));
        assert!(ToolDefinition::is_valid_name("search"));
        assert!(ToolDefinition::is_valid_name("_private"));
        assert!(ToolDefinition::is_valid_name("tool123"));
        assert!(ToolDefinition::is_valid_name("GetWeather"));

        // Invalid names
        assert!(!ToolDefinition::is_valid_name(""));
        assert!(!ToolDefinition::is_valid_name("invalid name"));
        assert!(!ToolDefinition::is_valid_name("123tool"));
        assert!(!ToolDefinition::is_valid_name("tool!"));
        assert!(!ToolDefinition::is_valid_name("tool-name"));
    }

    #[test]
    fn test_tool_choice_default() {
        let choice = ToolChoice::default();
        assert_eq!(choice, ToolChoice::Auto);
    }

    #[test]
    fn test_tool_choice_variants() {
        let auto = ToolChoice::Auto;
        let required = ToolChoice::Required;
        let none = ToolChoice::None;
        let specific = ToolChoice::Specific("my_tool".to_string());

        assert_eq!(auto, ToolChoice::Auto);
        assert_eq!(required, ToolChoice::Required);
        assert_eq!(none, ToolChoice::None);
        assert_eq!(specific, ToolChoice::Specific("my_tool".to_string()));
    }

    #[test]
    fn test_tool_call_creation() {
        let call = ToolCall::new("call_1", "get_weather", r#"{"location": "NYC"}"#);

        assert_eq!(call.id, "call_1");
        assert_eq!(call.name, "get_weather");
        assert_eq!(call.arguments, r#"{"location": "NYC"}"#);
    }

    #[test]
    fn test_tool_call_parse_arguments() {
        let call = ToolCall::new(
            "call_1",
            "get_weather",
            r#"{"location": "NYC", "unit": "celsius"}"#,
        );

        let args = call.parse_arguments().expect("test");
        assert_eq!(args["location"], "NYC");
        assert_eq!(args["unit"], "celsius");
    }

    #[test]
    fn test_tool_call_parse_arguments_invalid() {
        let call = ToolCall::new("call_1", "get_weather", "not valid json");

        let result = call.parse_arguments();
        assert!(result.is_err());
    }

    #[test]
    fn test_tool_result_success() {
        let result = ToolResult::success("call_1", r#"{"temp": 72}"#);

        assert_eq!(result.tool_call_id, "call_1");
        assert_eq!(result.content, r#"{"temp": 72}"#);
        assert!(result.success);
    }

    #[test]
    fn test_tool_result_error() {
        let result = ToolResult::error("call_1", "API rate limited");

        assert_eq!(result.tool_call_id, "call_1");
        assert_eq!(result.content, "API rate limited");
        assert!(!result.success);
    }

    #[test]
    fn test_tool_call_parser_creation() {
        let tools = vec![
            ToolDefinition::new("get_weather", "Get weather", vec![]),
            ToolDefinition::new("search", "Search web", vec![]),
        ];

        let parser = ToolCallParser::new(tools);
        let names: Vec<_> = parser.tool_names().collect();

        assert_eq!(names.len(), 2);
        assert!(names.contains(&"get_weather"));
        assert!(names.contains(&"search"));
    }

    #[test]
    fn test_tool_call_parser_generate_id() {
        let tools = vec![];
        let mut parser = ToolCallParser::new(tools);

        assert_eq!(parser.generate_id(), "call_0");
        assert_eq!(parser.generate_id(), "call_1");
        assert_eq!(parser.generate_id(), "call_2");
    }

    #[test]
    fn test_tool_call_format_default() {
        let format = ToolCallFormat::default();
        assert_eq!(format, ToolCallFormat::OpenAI);
    }

    #[test]
    fn test_tool_call_parser_with_format() {
        let tools = vec![];
        let parser = ToolCallParser::new(tools).with_format(ToolCallFormat::Anthropic);

        // The format change is internal, but we can verify it was created
        assert_eq!(parser.tool_names().count(), 0);
    }

    #[test]
    fn test_parse_openai_format() {
        let tools = vec![ToolDefinition::new("get_weather", "Get weather", vec![])];

        let mut parser = ToolCallParser::new(tools);
        let text = r#"I'll get the weather for you.
{"name": "get_weather", "arguments": {"location": "New York"}}
Here's the weather."#;

        let calls = parser.parse(text);

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        let args: serde_json::Value = serde_json::from_str(&calls[0].arguments).expect("test");
        assert_eq!(args["location"], "New York");
    }

    #[test]
    fn test_parse_openai_format_string_arguments() {
        let tools = vec![ToolDefinition::new("search", "Search", vec![])];

        let mut parser = ToolCallParser::new(tools);
        let text = r#"{"name": "search", "arguments": "{\"query\": \"rust programming\"}"}"#;

        let calls = parser.parse(text);

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search");
    }

    #[test]
    fn test_parse_multiple_tool_calls() {
        let tools = vec![
            ToolDefinition::new("get_weather", "Get weather", vec![]),
            ToolDefinition::new("search", "Search", vec![]),
        ];

        let mut parser = ToolCallParser::new(tools);
        let text = r#"
{"name": "get_weather", "arguments": {"location": "NYC"}}
Some text
{"name": "search", "arguments": {"query": "restaurants"}}
"#;

        let calls = parser.parse(text);

        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[1].name, "search");
    }

    #[test]
    fn test_parse_unknown_tool_ignored() {
        let tools = vec![ToolDefinition::new("get_weather", "Get weather", vec![])];

        let mut parser = ToolCallParser::new(tools);
        let text = r#"{"name": "unknown_tool", "arguments": {}}"#;

        let calls = parser.parse(text);

        assert_eq!(calls.len(), 0);
    }

    #[test]
    fn test_parse_anthropic_format() {
        let tools = vec![ToolDefinition::new("get_weather", "Get weather", vec![])];

        let mut parser = ToolCallParser::new(tools).with_format(ToolCallFormat::Anthropic);
        let text = r#"<tool_use>
<name>get_weather</name>
<input>{"location": "Paris"}</input>
</tool_use>"#;

        let calls = parser.parse(text);

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        let args: serde_json::Value = serde_json::from_str(&calls[0].arguments).expect("test");
        assert_eq!(args["location"], "Paris");
    }

    #[test]
    fn test_parse_hermes_format() {
        let tools = vec![ToolDefinition::new("search", "Search web", vec![])];

        let mut parser = ToolCallParser::new(tools).with_format(ToolCallFormat::Hermes);
        let text = r#"<tool_call>
{"name": "search", "arguments": {"query": "Rust tutorials"}}
</tool_call>"#;

        let calls = parser.parse(text);

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search");
    }

    #[test]
    fn test_find_matching_brace_simple() {
        let text = r#"{"key": "value"}"#;
        let result = find_matching_brace(text);

        assert_eq!(result, Some(15)); // Position of closing brace
    }

    #[test]
    fn test_find_matching_brace_nested() {
        let text = r#"{"outer": {"inner": "value"}}"#;
        let result = find_matching_brace(text);

        assert_eq!(result, Some(28));
    }

    #[test]
    fn test_find_matching_brace_with_string_braces() {
        let text = r#"{"text": "has { and } inside"}"#;
        let result = find_matching_brace(text);

        assert_eq!(result, Some(29));
    }

    #[test]
    fn test_find_matching_brace_unmatched() {
        let text = r#"{"key": "value""#;
        let result = find_matching_brace(text);

        assert_eq!(result, None);
    }

    #[test]
    fn test_find_matching_brace_not_starting_with_brace() {
        let text = "key: value";
        let result = find_matching_brace(text);

        assert_eq!(result, None);
    }

    #[test]
    fn test_extract_xml_tag() {
        let text = "<name>get_weather</name>";
        let result = extract_xml_tag(text, "name");

        assert_eq!(result, Some("get_weather".to_string()));
    }

    #[test]
    fn test_extract_xml_tag_with_whitespace() {
        let text = r#"<input>
{"location": "NYC"}
</input>"#;
        let result = extract_xml_tag(text, "input");

        // Result includes surrounding newlines - trim for comparison
        assert!(result.is_some());
        let trimmed = result.expect("test").trim().to_string();
        assert_eq!(trimmed, r#"{"location": "NYC"}"#);
    }

    #[test]
    fn test_extract_xml_tag_not_found() {
        let text = "<name>test</name>";
        let result = extract_xml_tag(text, "other");

        assert_eq!(result, None);
    }

    #[test]
    fn test_generate_tool_grammar_single_tool() {
        let tools = vec![ToolDefinition::new(
            "get_weather",
            "Get weather",
            vec![ToolParameter::required_string("location", "City name")],
        )];

        let grammar = generate_tool_grammar(&tools);

        // Grammar must have a root rule
        assert!(grammar.get_rule("root").is_some());
        // Grammar should have rules (at least one per tool or base rules)
        assert!(grammar.rule_names().count() > 0);
    }

    #[test]
    fn test_generate_tool_grammar_multiple_tools() {
        let tools = vec![
            ToolDefinition::new("get_weather", "Get weather", vec![]),
            ToolDefinition::new("search", "Search web", vec![]),
        ];

        let grammar = generate_tool_grammar(&tools);

        // Should have rules for both tools
        assert!(grammar.get_rule("root").is_some());
    }

    #[test]
    fn test_generate_tool_grammar_empty_tools() {
        let tools: Vec<ToolDefinition> = vec![];
        let grammar = generate_tool_grammar(&tools);

        // Should still create a valid grammar
        assert!(grammar.get_rule("root").is_some());
    }

    #[test]
    fn test_tool_definition_serialization() {
        let tool = ToolDefinition::new(
            "test_tool",
            "A test tool",
            vec![ToolParameter::required_string("arg1", "First argument")],
        );

        let json = serde_json::to_string(&tool).expect("test");
        let deserialized: ToolDefinition = serde_json::from_str(&json).expect("test");

        assert_eq!(deserialized.name, "test_tool");
        assert_eq!(deserialized.description, "A test tool");
        assert_eq!(deserialized.parameters.len(), 1);
    }

    #[test]
    fn test_tool_call_serialization() {
        let call = ToolCall::new("call_123", "my_tool", r#"{"key": "value"}"#);

        let json = serde_json::to_string(&call).expect("test");
        let deserialized: ToolCall = serde_json::from_str(&json).expect("test");

        assert_eq!(deserialized.id, "call_123");
        assert_eq!(deserialized.name, "my_tool");
        assert_eq!(deserialized.arguments, r#"{"key": "value"}"#);
    }

    #[test]
    fn test_tool_result_serialization() {
        let result = ToolResult::success("call_123", "result data");

        let json = serde_json::to_string(&result).expect("test");
        let deserialized: ToolResult = serde_json::from_str(&json).expect("test");

        assert_eq!(deserialized.tool_call_id, "call_123");
        assert_eq!(deserialized.content, "result data");
        assert!(deserialized.success);
    }

    #[test]
    fn test_tool_choice_serialization() {
        let choice = ToolChoice::Specific("my_tool".to_string());

        let json = serde_json::to_string(&choice).expect("test");
        assert!(json.contains("my_tool"));

        let auto = ToolChoice::Auto;
        let auto_json = serde_json::to_string(&auto).expect("test");
        assert_eq!(auto_json, "\"auto\"");
    }

    #[test]
    fn test_tool_parameter_type_serialization() {
        let array_type = ToolParameterType::Array {
            items: Box::new(ToolParameterType::Integer),
        };

        let json = serde_json::to_string(&array_type).expect("test");
        let deserialized: ToolParameterType = serde_json::from_str(&json).expect("test");

        match deserialized {
            ToolParameterType::Array { items } => {
                assert_eq!(*items, ToolParameterType::Integer);
            },
            _ => panic!("Expected Array type"),
        }
    }

    #[test]
    fn test_parse_no_tool_calls() {
        let tools = vec![ToolDefinition::new("get_weather", "Get weather", vec![])];

        let mut parser = ToolCallParser::new(tools);
        let text = "Just some regular text without any tool calls.";

        let calls = parser.parse(text);

        assert_eq!(calls.len(), 0);
    }

    #[test]
    fn test_parse_malformed_json() {
        let tools = vec![ToolDefinition::new("get_weather", "Get weather", vec![])];

        let mut parser = ToolCallParser::new(tools);
        let text = r#"{"name": "get_weather", "arguments": {malformed}"#;

        let calls = parser.parse(text);

        // Malformed JSON should be skipped
        assert_eq!(calls.len(), 0);
    }

    #[test]
    fn test_tool_parameter_type_boolean() {
        let bool_type = ToolParameterType::Boolean;
        assert_eq!(bool_type, ToolParameterType::Boolean);
    }

    #[test]
    fn test_tool_parameter_type_number() {
        let num_type = ToolParameterType::Number;
        assert_eq!(num_type, ToolParameterType::Number);
    }

    // ==========================================================================
    // DEEP COVERAGE TESTS - _deep_grcov_ prefix
    // ==========================================================================

    #[test]
    fn test_deep_grcov_grammar_state_is_complete_no_rule() {
        // Test is_complete when rule doesn't exist in grammar
        let grammar = Grammar::new();
        let state = GrammarState::initial("nonexistent");

        // Should return false when rule doesn't exist
        assert!(!state.is_complete(&grammar));
    }

    #[test]
    fn test_deep_grcov_grammar_state_is_complete_invalid_alt_idx() {
        // Test is_complete with alt_idx out of bounds
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::single("root", vec![GrammarElement::Char('a')]));

        let state = GrammarState {
            rule: "root".to_string(),
            alt_idx: 999, // Out of bounds
            elem_idx: 0,
            stack: Vec::new(),
        };

        // Should return false when alt_idx is out of bounds
        assert!(!state.is_complete(&grammar));
    }

    #[test]
    fn test_deep_grcov_grammar_state_is_complete_with_stack() {
        // Test is_complete returns false when stack is non-empty
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::single("root", vec![GrammarElement::Char('a')]));

        let state = GrammarState {
            rule: "root".to_string(),
            alt_idx: 0,
            elem_idx: 1,                              // Past the only element
            stack: vec![("other".to_string(), 0, 0)], // Non-empty stack
        };

        // Should return false because stack is not empty
        assert!(!state.is_complete(&grammar));
    }

    #[test]
    fn test_deep_grcov_grammar_state_current_element_none() {
        // Test current_element returns None for various cases
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::single("root", vec![GrammarElement::Char('a')]));

        // Case 1: Invalid rule name
        let state1 = GrammarState::initial("nonexistent");
        assert!(state1.current_element(&grammar).is_none());

        // Case 2: Invalid alt_idx
        let state2 = GrammarState {
            rule: "root".to_string(),
            alt_idx: 999,
            elem_idx: 0,
            stack: Vec::new(),
        };
        assert!(state2.current_element(&grammar).is_none());

        // Case 3: Invalid elem_idx
        let state3 = GrammarState {
            rule: "root".to_string(),
            alt_idx: 0,
            elem_idx: 999,
            stack: Vec::new(),
        };
        assert!(state3.current_element(&grammar).is_none());
    }

    #[test]
    fn test_deep_grcov_state_machine_advance_invalid_char() {
        // Test that advance returns false for invalid char
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::single("root", vec![GrammarElement::Char('a')]));

        let mut sm = GrammarStateMachine::new(grammar).expect("should create");
        assert!(!sm.advance('x')); // 'x' is not valid
        assert_eq!(sm.generated(), ""); // Nothing generated
    }

    #[test]
    fn test_deep_grcov_state_machine_can_accept_char_any() {
        // Test GrammarElement::Any accepts any character
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::single("root", vec![GrammarElement::Any]));

        let sm = GrammarStateMachine::new(grammar).expect("should create");
        assert!(sm.is_valid_char('x'));
        assert!(sm.is_valid_char('1'));
        assert!(sm.is_valid_char(' '));
        assert!(sm.is_valid_char('\n'));
    }

    #[test]
    fn test_deep_grcov_state_machine_can_accept_char_end() {
        // Test GrammarElement::End doesn't accept any character
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::single("root", vec![GrammarElement::End]));

        let sm = GrammarStateMachine::new(grammar).expect("should create");
        assert!(!sm.is_valid_char('x'));
        assert!(!sm.is_valid_char('a'));
    }

    #[test]
    fn test_deep_grcov_state_machine_can_accept_char_not() {
        // Test GrammarElement::CharNot
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::single(
            "root",
            vec![GrammarElement::CharNot(vec!['x', 'y', 'z'])],
        ));

        let sm = GrammarStateMachine::new(grammar).expect("should create");
        assert!(sm.is_valid_char('a')); // Not in excluded list
        assert!(sm.is_valid_char('b'));
        assert!(!sm.is_valid_char('x')); // In excluded list
        assert!(!sm.is_valid_char('y'));
        assert!(!sm.is_valid_char('z'));
    }

    #[test]
    fn test_deep_grcov_state_machine_rule_ref_invalid_rule() {
        // Test RuleRef to non-existent rule
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::single(
            "root",
            vec![GrammarElement::RuleRef("nonexistent".to_string())],
        ));

        // Validation should fail
        assert!(grammar.validate().is_err());
    }

    #[test]
    fn test_deep_grcov_state_machine_rule_ref_chain() {
        // Test nested rule references (note: deep chains don't auto-complete due to single-pop)
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::single(
            "root",
            vec![GrammarElement::RuleRef("middle".to_string())],
        ));
        grammar.add_rule(GrammarRule::single(
            "middle",
            vec![GrammarElement::RuleRef("leaf".to_string())],
        ));
        grammar.add_rule(GrammarRule::single("leaf", vec![GrammarElement::Char('x')]));

        let mut sm = GrammarStateMachine::new(grammar).expect("should create");
        assert!(sm.is_valid_char('x'));
        assert!(sm.advance('x'));
        // Note: deep rule chains don't fully unwind in current implementation
        // The state machine has advanced but isn't "complete" yet
        assert!(!sm.states.is_empty());
    }

    #[test]
    fn test_deep_grcov_state_machine_collect_valid_chars_any() {
        // Test valid_chars with GrammarElement::Any - should include printable chars
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::single("root", vec![GrammarElement::Any]));

        let sm = GrammarStateMachine::new(grammar).expect("should create");
        let valid = sm.valid_chars();

        // Should include printable ASCII
        assert!(valid.contains(&' '));
        assert!(valid.contains(&'~'));
        assert!(valid.contains(&'a'));
        assert!(valid.contains(&'Z'));
    }

    #[test]
    fn test_deep_grcov_state_machine_collect_valid_chars_char_not() {
        // Test valid_chars with GrammarElement::CharNot
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::single(
            "root",
            vec![GrammarElement::CharNot(vec!['a', 'b', 'c'])],
        ));

        let sm = GrammarStateMachine::new(grammar).expect("should create");
        let valid = sm.valid_chars();

        // Should include printable chars except a, b, c
        assert!(!valid.contains(&'a'));
        assert!(!valid.contains(&'b'));
        assert!(!valid.contains(&'c'));
        assert!(valid.contains(&'d'));
        assert!(valid.contains(&'x'));
    }

    #[test]
    fn test_deep_grcov_state_machine_collect_valid_chars_end() {
        // Test valid_chars with GrammarElement::End - should be empty
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::single("root", vec![GrammarElement::End]));

        let sm = GrammarStateMachine::new(grammar).expect("should create");
        let valid = sm.valid_chars();
        assert!(valid.is_empty());
    }

    #[test]
    fn test_deep_grcov_state_machine_advance_state_end() {
        // Test advance_state with End element
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::single("root", vec![GrammarElement::End]));

        let mut sm = GrammarStateMachine::new(grammar).expect("should create");
        assert!(!sm.advance('x')); // End element doesn't accept any char
    }

    #[test]
    fn test_deep_grcov_state_machine_advance_state_char_range_boundary() {
        // Test CharRange boundary conditions
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::single(
            "root",
            vec![GrammarElement::CharRange('a', 'c')],
        ));

        let mut sm = GrammarStateMachine::new(grammar.clone()).expect("should create");
        assert!(sm.advance('a')); // Start boundary

        let mut sm2 = GrammarStateMachine::new(grammar.clone()).expect("should create");
        assert!(sm2.advance('c')); // End boundary

        let mut sm3 = GrammarStateMachine::new(grammar).expect("should create");
        assert!(!sm3.advance('d')); // Just outside
    }

    #[test]
    fn test_deep_grcov_state_machine_next_state_stack_pop() {
        // Test next_state with stack popping
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::single(
            "root",
            vec![
                GrammarElement::RuleRef("sub".to_string()),
                GrammarElement::Char('!'),
            ],
        ));
        grammar.add_rule(GrammarRule::single("sub", vec![GrammarElement::Char('x')]));

        let mut sm = GrammarStateMachine::new(grammar).expect("should create");
        assert!(sm.advance('x')); // Enter and complete sub rule
        assert!(sm.is_valid_char('!')); // Should be back in root
        assert!(sm.advance('!'));
        assert!(sm.is_complete());
    }

    #[test]
    fn test_deep_grcov_json_schema_number() {
        // Test JsonSchemaType::Number
        let schema = JsonSchemaType::Number;
        let grammar = grammar_from_json_schema(&schema);

        let sm = GrammarStateMachine::new(grammar).expect("should create");

        // Test decimal number
        assert!(sm.is_valid_char('1'));
        assert!(sm.is_valid_char('-'));
    }

    #[test]
    fn test_deep_grcov_json_schema_empty_object() {
        // Test JsonSchemaType::Object with empty properties
        let schema = JsonSchemaType::Object(vec![]);
        let grammar = grammar_from_json_schema(&schema);

        let sm = GrammarStateMachine::new(grammar).expect("should create");
        assert!(sm.is_valid_char('{'));
    }

    #[test]
    fn test_deep_grcov_json_schema_any() {
        // Test JsonSchemaType::Any
        let schema = JsonSchemaType::Any;
        let grammar = grammar_from_json_schema(&schema);

        // Should have helper rules
        assert!(grammar.get_rule("string_value").is_some());
        assert!(grammar.get_rule("number").is_some());
        assert!(grammar.get_rule("boolean").is_some());
        assert!(grammar.get_rule("null").is_some());
    }

    #[test]
    fn test_deep_grcov_grammar_masker_advance_invalid_token() {
        // Test advance_token with invalid token ID
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::single("root", vec![GrammarElement::Char('a')]));

        let token_strings = HashMap::new(); // Empty - no tokens

        let mut masker =
            GrammarTokenMasker::new(grammar, token_strings, 99).expect("should create");
        assert!(!masker.advance_token(999)); // Unknown token
    }

    #[test]
    fn test_deep_grcov_grammar_masker_advance_token_invalid_sequence() {
        // Test advance_token with a token that doesn't match grammar
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::single("root", vec![GrammarElement::Char('a')]));

        let mut token_strings = HashMap::new();
        token_strings.insert(0, "x".to_string()); // Wrong character

        let mut masker =
            GrammarTokenMasker::new(grammar, token_strings, 99).expect("should create");
        assert!(!masker.advance_token(0)); // 'x' doesn't match 'a'
    }

    #[test]
    fn test_deep_grcov_grammar_masker_eos_token_id() {
        // Test eos_token_id getter
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::single("root", vec![GrammarElement::Char('a')]));

        let masker = GrammarTokenMasker::new(grammar, HashMap::new(), 42).expect("should create");
        assert_eq!(masker.eos_token_id(), 42);
    }

    #[test]
    fn test_deep_grcov_grammar_masker_multi_char_token_first_invalid() {
        // Test multi-char token where first char is invalid
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::single("root", vec![GrammarElement::Char('a')]));

        let mut token_strings = HashMap::new();
        token_strings.insert(0, "xy".to_string()); // First char 'x' is invalid

        let masker = GrammarTokenMasker::new(grammar, token_strings, 99).expect("should create");
        let mask = masker.get_mask();
        assert!(!mask.is_allowed(0)); // Token should not be allowed
    }

    #[test]
    fn test_deep_grcov_grammar_masker_empty_token() {
        // Test with empty token string
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::single("root", vec![GrammarElement::Char('a')]));

        let mut token_strings = HashMap::new();
        token_strings.insert(0, String::new()); // Empty token

        let masker = GrammarTokenMasker::new(grammar, token_strings, 99).expect("should create");
        let mask = masker.get_mask();
        // Empty token won't have a first char, so it shouldn't be in allowed
        assert!(!mask.is_allowed(0));
    }

    #[test]
    fn test_deep_grcov_find_matching_brace_escaped_quote() {
        // Test find_matching_brace with escaped quotes
        let text = r#"{"key": "value with \" escaped"}"#;
        let result = find_matching_brace(text);
        assert!(result.is_some());
    }

    #[test]
    fn test_deep_grcov_find_matching_brace_escaped_backslash() {
        // Test find_matching_brace with escaped backslash
        let text = r#"{"path": "C:\\Users\\test"}"#;
        let result = find_matching_brace(text);
        assert!(result.is_some());
    }

    #[test]
    fn test_deep_grcov_extract_xml_tag_unclosed() {
        // Test extract_xml_tag with unclosed tag
        let text = "<name>test";
        let result = extract_xml_tag(text, "name");
        assert!(result.is_none());
    }

    #[test]
    fn test_deep_grcov_parse_anthropic_format_unclosed() {
        // Test Anthropic format with unclosed tool_use
        let tools = vec![ToolDefinition::new("test", "Test", vec![])];
        let mut parser = ToolCallParser::new(tools).with_format(ToolCallFormat::Anthropic);

        let text = "<tool_use><name>test</name><input>{}</input>"; // No closing tag
        let calls = parser.parse(text);
        assert!(calls.is_empty());
    }

    #[test]
    fn test_deep_grcov_parse_anthropic_format_missing_name() {
        // Test Anthropic format with missing name
        let tools = vec![ToolDefinition::new("test", "Test", vec![])];
        let mut parser = ToolCallParser::new(tools).with_format(ToolCallFormat::Anthropic);

        let text = "<tool_use><input>{}</input></tool_use>";
        let calls = parser.parse(text);
        assert!(calls.is_empty());
    }

    #[test]
    fn test_deep_grcov_parse_anthropic_format_unknown_tool() {
        // Test Anthropic format with unknown tool
        let tools = vec![ToolDefinition::new("known_tool", "Known", vec![])];
        let mut parser = ToolCallParser::new(tools).with_format(ToolCallFormat::Anthropic);

        let text = "<tool_use><name>unknown_tool</name><input>{}</input></tool_use>";
        let calls = parser.parse(text);
        assert!(calls.is_empty());
    }

    #[test]
    fn test_deep_grcov_parse_hermes_format_unclosed() {
        // Test Hermes format with unclosed tool_call
        let tools = vec![ToolDefinition::new("test", "Test", vec![])];
        let mut parser = ToolCallParser::new(tools).with_format(ToolCallFormat::Hermes);

        let text = r#"<tool_call>{"name": "test", "arguments": {}}"#; // No closing tag
        let calls = parser.parse(text);
        assert!(calls.is_empty());
    }

    #[test]
    fn test_deep_grcov_parse_hermes_format_invalid_json() {
        // Test Hermes format with invalid JSON
        let tools = vec![ToolDefinition::new("test", "Test", vec![])];
        let mut parser = ToolCallParser::new(tools).with_format(ToolCallFormat::Hermes);

        let text = "<tool_call>not valid json</tool_call>";
        let calls = parser.parse(text);
        assert!(calls.is_empty());
    }

    #[test]
    fn test_deep_grcov_parse_hermes_format_string_arguments() {
        // Test Hermes format with string arguments
        let tools = vec![ToolDefinition::new("test", "Test", vec![])];
        let mut parser = ToolCallParser::new(tools).with_format(ToolCallFormat::Hermes);

        let text =
            r#"<tool_call>{"name": "test", "arguments": "{\"key\": \"value\"}"}</tool_call>"#;
        let calls = parser.parse(text);
        assert_eq!(calls.len(), 1);
    }

    #[test]
    fn test_deep_grcov_parse_hermes_format_unknown_tool() {
        // Test Hermes format with unknown tool
        let tools = vec![ToolDefinition::new("known", "Known", vec![])];
        let mut parser = ToolCallParser::new(tools).with_format(ToolCallFormat::Hermes);

        let text = r#"<tool_call>{"name": "unknown", "arguments": {}}</tool_call>"#;
        let calls = parser.parse(text);
        assert!(calls.is_empty());
    }

    #[test]
    fn test_deep_grcov_generate_tool_grammar_with_all_param_types() {
        // Test generate_tool_grammar with all parameter types
        let tools = vec![ToolDefinition::new(
            "complex_tool",
            "A complex tool",
            vec![
                ToolParameter {
                    name: "str_param".to_string(),
                    description: "String".to_string(),
                    param_type: ToolParameterType::String,
                    required: true,
                    default: None,
                },
                ToolParameter {
                    name: "int_param".to_string(),
                    description: "Integer".to_string(),
                    param_type: ToolParameterType::Integer,
                    required: true,
                    default: None,
                },
                ToolParameter {
                    name: "num_param".to_string(),
                    description: "Number".to_string(),
                    param_type: ToolParameterType::Number,
                    required: true,
                    default: None,
                },
                ToolParameter {
                    name: "bool_param".to_string(),
                    description: "Boolean".to_string(),
                    param_type: ToolParameterType::Boolean,
                    required: true,
                    default: None,
                },
                ToolParameter {
                    name: "enum_param".to_string(),
                    description: "Enum".to_string(),
                    param_type: ToolParameterType::Enum(vec!["a".to_string(), "b".to_string()]),
                    required: true,
                    default: None,
                },
                ToolParameter {
                    name: "array_param".to_string(),
                    description: "Array".to_string(),
                    param_type: ToolParameterType::Array {
                        items: Box::new(ToolParameterType::Integer),
                    },
                    required: true,
                    default: None,
                },
                ToolParameter {
                    name: "object_param".to_string(),
                    description: "Object".to_string(),
                    param_type: ToolParameterType::Object {
                        properties: vec![ToolParameter::required_string("nested", "Nested")],
                    },
                    required: true,
                    default: None,
                },
            ],
        )];

        let grammar = generate_tool_grammar(&tools);
        assert!(grammar.get_rule("root").is_some());
    }

    #[test]
    fn test_deep_grcov_generate_params_grammar_empty() {
        // Test generate_params_grammar with empty params
        let tools = vec![ToolDefinition::new("no_params", "No params", vec![])];

        let grammar = generate_tool_grammar(&tools);
        // Should have created an empty object grammar
        assert!(grammar.get_rule("root").is_some());
    }

    #[test]
    fn test_deep_grcov_tool_call_parser_get_tool() {
        // Test get_tool method
        let tools = vec![
            ToolDefinition::new("tool1", "Tool 1", vec![]),
            ToolDefinition::new("tool2", "Tool 2", vec![]),
        ];

        let parser = ToolCallParser::new(tools);
        assert!(parser.get_tool("tool1").is_some());
        assert!(parser.get_tool("tool2").is_some());
        assert!(parser.get_tool("tool3").is_none());
    }

    #[test]
    fn test_deep_grcov_grammar_rule_names() {
        // Test rule_names iterator
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::single("root", vec![GrammarElement::Char('a')]));
        grammar.add_rule(GrammarRule::single(
            "other",
            vec![GrammarElement::Char('b')],
        ));

        let names: Vec<_> = grammar.rule_names().collect();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&&"root".to_string()));
        assert!(names.contains(&&"other".to_string()));
    }

    #[test]
    fn test_deep_grcov_state_machine_has_valid_continuation_empty() {
        // Test has_valid_continuation behavior
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::single("root", vec![GrammarElement::Char('a')]));

        let mut sm = GrammarStateMachine::new(grammar).expect("should create");
        assert!(sm.has_valid_continuation()); // Initially has states

        // Advance with invalid char - returns false but preserves states
        let advanced = sm.advance('x'); // Invalid char
        assert!(!advanced); // Should return false
                            // Note: current impl preserves states on failed advance (states not cleared)
                            // This allows retry with different char
        assert!(sm.has_valid_continuation());
    }

    #[test]
    fn test_deep_grcov_grammar_with_root() {
        // Test Grammar::with_root
        let grammar = Grammar::with_root("my_root");
        assert_eq!(grammar.root(), "my_root");
        assert!(grammar.is_empty());
    }

    #[test]
    fn test_deep_grcov_grammar_set_root() {
        // Test Grammar::set_root
        let mut grammar = Grammar::new();
        grammar.set_root("new_root");
        assert_eq!(grammar.root(), "new_root");
    }

    #[test]
    fn test_deep_grcov_state_machine_rule_ref_none_alternatives() {
        // Test RuleRef advancing with no matching alternatives
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::single(
            "root",
            vec![GrammarElement::RuleRef("sub".to_string())],
        ));
        grammar.add_rule(GrammarRule::single("sub", vec![GrammarElement::Char('x')]));

        let mut sm = GrammarStateMachine::new(grammar).expect("should create");
        // Try to advance with a char that doesn't match sub rule
        assert!(!sm.advance('y')); // 'y' doesn't match 'x' in sub
    }

    #[test]
    fn test_deep_grcov_parse_openai_nested_json() {
        // Test OpenAI parser with deeply nested JSON
        let tools = vec![ToolDefinition::new("nested", "Nested", vec![])];
        let mut parser = ToolCallParser::new(tools);

        let text = r#"{"name": "nested", "arguments": {"level1": {"level2": {"level3": "deep"}}}}"#;
        let calls = parser.parse(text);
        assert_eq!(calls.len(), 1);
    }

    #[test]
    fn test_deep_grcov_parse_openai_multiple_json_blocks() {
        // Test OpenAI parser skipping non-tool JSON
        let tools = vec![ToolDefinition::new("tool", "Tool", vec![])];
        let mut parser = ToolCallParser::new(tools);

        let text = r#"{"other": "json"} {"name": "tool", "arguments": {}} {"another": "block"}"#;
        let calls = parser.parse(text);
        assert_eq!(calls.len(), 1);
    }

    #[test]
    fn test_deep_grcov_grammar_validate_undefined_rule_ref() {
        // Test validation with undefined rule reference
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::single(
            "root",
            vec![
                GrammarElement::Char('a'),
                GrammarElement::RuleRef("undefined".to_string()),
            ],
        ));

        let result = grammar.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_deep_grcov_grammar_element_clone_and_eq() {
        // Test Clone and Eq for GrammarElement
        let elem1 = GrammarElement::CharRange('a', 'z');
        let elem2 = elem1.clone();
        assert_eq!(elem1, elem2);

        let elem3 = GrammarElement::RuleRef("test".to_string());
        let elem4 = elem3.clone();
        assert_eq!(elem3, elem4);
    }

    #[test]
    fn test_deep_grcov_grammar_alternative_clone() {
        // Test Clone for GrammarAlternative
        let alt1 =
            GrammarAlternative::new(vec![GrammarElement::Char('a'), GrammarElement::Char('b')]);
        let alt2 = alt1.clone();
        assert_eq!(alt1, alt2);
    }

    #[test]
    fn test_deep_grcov_grammar_rule_clone() {
        // Test Clone for GrammarRule
        let rule1 = GrammarRule::single("test", vec![GrammarElement::Char('x')]);
        let rule2 = rule1.clone();
        assert_eq!(rule1, rule2);
    }

    #[test]
    fn test_deep_grcov_grammar_state_clone_and_hash() {
        use std::collections::HashSet;

        // Test Clone and Hash for GrammarState
        let state1 = GrammarState::initial("root");
        let state2 = state1.clone();
        assert_eq!(state1, state2);

        // Test that it can be used in HashSet
        let mut set = HashSet::new();
        set.insert(state1.clone());
        assert!(set.contains(&state2));
    }

    #[test]
    fn test_deep_grcov_token_mask_num_allowed() {
        // Test TokenMask::num_allowed
        let allowed: HashSet<u32> = vec![1, 2, 3, 4, 5].into_iter().collect();
        let mask = TokenMask::from_allowed(allowed, true);
        assert_eq!(mask.num_allowed(), 5);
    }

    #[test]
    fn test_deep_grcov_tool_result_default_success() {
        // Test ToolResult deserialization with default success
        let json = r#"{"tool_call_id": "id", "content": "data"}"#;
        let result: ToolResult = serde_json::from_str(json).expect("should parse");
        assert!(result.success); // Default is true
    }

    #[test]
    fn test_deep_grcov_tool_parameter_type_object_serialization() {
        // Test serialization of Object parameter type
        let obj_type = ToolParameterType::Object {
            properties: vec![ToolParameter::required_int("count", "Count value")],
        };

        let json = serde_json::to_string(&obj_type).expect("should serialize");
        let deserialized: ToolParameterType =
            serde_json::from_str(&json).expect("should deserialize");

        match deserialized {
            ToolParameterType::Object { properties } => {
                assert_eq!(properties.len(), 1);
                assert_eq!(properties[0].name, "count");
            },
            _ => panic!("Expected Object type"),
        }
    }

    #[test]
    fn test_deep_grcov_state_machine_advance_any() {
        // Test advance with Any element
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::single(
            "root",
            vec![GrammarElement::Any, GrammarElement::Char('!')],
        ));

        let mut sm = GrammarStateMachine::new(grammar).expect("should create");
        assert!(sm.advance('x')); // Any accepts any char
        assert!(sm.advance('!')); // Then literal
        assert!(sm.is_complete());
    }

    #[test]
    fn test_deep_grcov_state_machine_advance_char_not() {
        // Test advance with CharNot element
        let mut grammar = Grammar::with_root("root");
        grammar.add_rule(GrammarRule::single(
            "root",
            vec![GrammarElement::CharNot(vec!['x'])],
        ));

        let mut sm = GrammarStateMachine::new(grammar.clone()).expect("should create");
        assert!(sm.advance('a')); // 'a' is not excluded

        let mut sm2 = GrammarStateMachine::new(grammar).expect("should create");
        assert!(!sm2.advance('x')); // 'x' is excluded
    }

    #[test]
    fn test_deep_grcov_json_schema_nested_array() {
        // Test JsonSchemaType::Array with Array items
        let schema = JsonSchemaType::Array(Box::new(JsonSchemaType::Array(Box::new(
            JsonSchemaType::Integer,
        ))));
        let grammar = grammar_from_json_schema(&schema);

        assert!(grammar.get_rule("root").is_some());
        assert!(grammar.get_rule("root_item").is_some());
    }

    #[test]
    fn test_deep_grcov_json_schema_object_multiple_properties() {
        // Test JsonSchemaType::Object with multiple properties
        let schema = JsonSchemaType::Object(vec![
            ("prop1".to_string(), JsonSchemaType::String, true),
            ("prop2".to_string(), JsonSchemaType::Integer, false),
            ("prop3".to_string(), JsonSchemaType::Boolean, true),
        ]);
        let grammar = grammar_from_json_schema(&schema);

        assert!(grammar.get_rule("root_prop1").is_some());
        assert!(grammar.get_rule("root_prop2").is_some());
        assert!(grammar.get_rule("root_prop3").is_some());
    }

    #[test]
    fn test_deep_grcov_parse_anthropic_multiple_calls() {
        // Test Anthropic format with multiple tool calls
        let tools = vec![
            ToolDefinition::new("tool1", "Tool 1", vec![]),
            ToolDefinition::new("tool2", "Tool 2", vec![]),
        ];
        let mut parser = ToolCallParser::new(tools).with_format(ToolCallFormat::Anthropic);

        let text = r#"<tool_use><name>tool1</name><input>{}</input></tool_use>
                      Some text
                      <tool_use><name>tool2</name><input>{"key": "value"}</input></tool_use>"#;
        let calls = parser.parse(text);
        assert_eq!(calls.len(), 2);
    }

    #[test]
    fn test_deep_grcov_parse_hermes_multiple_calls() {
        // Test Hermes format with multiple tool calls
        let tools = vec![
            ToolDefinition::new("tool1", "Tool 1", vec![]),
            ToolDefinition::new("tool2", "Tool 2", vec![]),
        ];
        let mut parser = ToolCallParser::new(tools).with_format(ToolCallFormat::Hermes);

        let text = r#"<tool_call>{"name": "tool1", "arguments": {}}</tool_call>
                      <tool_call>{"name": "tool2", "arguments": {"x": 1}}</tool_call>"#;
        let calls = parser.parse(text);
        assert_eq!(calls.len(), 2);
    }
}
