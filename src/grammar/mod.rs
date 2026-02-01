//! Grammar-constrained generation for structured output
//!
//! Implements GBNF-style grammar constraints for LLM generation.
//! Supports JSON schema validation and custom grammar rules.
//!
//! Reference: llama.cpp grammar implementation
//! - GBNF format: Backus-Naur Form with extensions
//! - Token masking: Efficiently filter invalid tokens
//! - State machine: Track grammar state during generation

use crate::error::{RealizarError, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// =============================================================================
// GRAMMAR RULE TYPES
// =============================================================================

/// Grammar rule element types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GrammarElement {
    /// Literal character
    Char(char),
    /// Character range [a-z]
    CharRange(char, char),
    /// Reference to another rule
    RuleRef(String),
    /// Negated character set [^...]
    CharNot(Vec<char>),
    /// Any character
    Any,
    /// End of rule
    End,
}

/// Alternative production in a rule
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GrammarAlternative {
    /// Sequence of elements in this alternative
    pub elements: Vec<GrammarElement>,
}

impl GrammarAlternative {
    /// Create new alternative from elements
    pub fn new(elements: Vec<GrammarElement>) -> Self {
        Self { elements }
    }

    /// Create single-character alternative
    pub fn char(c: char) -> Self {
        Self {
            elements: vec![GrammarElement::Char(c)],
        }
    }

    /// Check if this alternative is empty (epsilon production)
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }
}

/// Grammar rule with one or more alternatives
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GrammarRule {
    /// Rule name
    pub name: String,
    /// Alternative productions
    pub alternatives: Vec<GrammarAlternative>,
}

impl GrammarRule {
    /// Create new rule with given alternatives
    pub fn new(name: impl Into<String>, alternatives: Vec<GrammarAlternative>) -> Self {
        Self {
            name: name.into(),
            alternatives,
        }
    }

    /// Create rule with single alternative
    pub fn single(name: impl Into<String>, elements: Vec<GrammarElement>) -> Self {
        Self {
            name: name.into(),
            alternatives: vec![GrammarAlternative::new(elements)],
        }
    }
}

// =============================================================================
// GRAMMAR DEFINITION
// =============================================================================

/// Complete grammar definition
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Grammar {
    /// Rules by name
    rules: HashMap<String, GrammarRule>,
    /// Root rule name
    root: String,
}

impl Grammar {
    /// Create empty grammar
    pub fn new() -> Self {
        Self {
            rules: HashMap::new(),
            root: String::new(),
        }
    }

    /// Create grammar with root rule
    pub fn with_root(root: impl Into<String>) -> Self {
        Self {
            rules: HashMap::new(),
            root: root.into(),
        }
    }

    /// Add a rule to the grammar
    pub fn add_rule(&mut self, rule: GrammarRule) {
        if self.root.is_empty() {
            self.root.clone_from(&rule.name);
        }
        self.rules.insert(rule.name.clone(), rule);
    }

    /// Get rule by name
    pub fn get_rule(&self, name: &str) -> Option<&GrammarRule> {
        self.rules.get(name)
    }

    /// Get root rule name
    pub fn root(&self) -> &str {
        &self.root
    }

    /// Set root rule
    pub fn set_root(&mut self, root: impl Into<String>) {
        self.root = root.into();
    }

    /// Get all rule names
    pub fn rule_names(&self) -> impl Iterator<Item = &String> {
        self.rules.keys()
    }

    /// Number of rules
    pub fn len(&self) -> usize {
        self.rules.len()
    }

    /// Check if grammar is empty
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// Validate grammar has required rules
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfiguration` if:
    /// - Grammar has no root rule
    /// - Root rule is not defined in the grammar
    /// - Any rule references an undefined rule
    pub fn validate(&self) -> Result<()> {
        if self.root.is_empty() {
            return Err(RealizarError::InvalidConfiguration(
                "Grammar has no root rule".to_string(),
            ));
        }

        if !self.rules.contains_key(&self.root) {
            return Err(RealizarError::InvalidConfiguration(format!(
                "Root rule '{}' not found in grammar",
                self.root
            )));
        }

        // Check all rule references are valid
        for rule in self.rules.values() {
            for alt in &rule.alternatives {
                for elem in &alt.elements {
                    if let GrammarElement::RuleRef(ref_name) = elem {
                        if !self.rules.contains_key(ref_name) {
                            return Err(RealizarError::InvalidConfiguration(format!(
                                "Rule '{}' references undefined rule '{}'",
                                rule.name, ref_name
                            )));
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

// =============================================================================
// GRAMMAR STATE MACHINE
// =============================================================================

/// State in the grammar state machine
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GrammarState {
    /// Current rule being matched
    pub rule: String,
    /// Alternative index within rule
    pub alt_idx: usize,
    /// Position within alternative
    pub elem_idx: usize,
    /// Stack of parent states (for rule references)
    pub stack: Vec<(String, usize, usize)>,
}

impl GrammarState {
    /// Create initial state from root rule
    pub fn initial(root: &str) -> Self {
        Self {
            rule: root.to_string(),
            alt_idx: 0,
            elem_idx: 0,
            stack: Vec::new(),
        }
    }

    /// Check if state is at end of rule
    pub fn is_complete(&self, grammar: &Grammar) -> bool {
        if let Some(rule) = grammar.get_rule(&self.rule) {
            if self.alt_idx < rule.alternatives.len() {
                let alt = &rule.alternatives[self.alt_idx];
                return self.elem_idx >= alt.elements.len() && self.stack.is_empty();
            }
        }
        false
    }

    /// Get current element being matched
    pub fn current_element<'a>(&self, grammar: &'a Grammar) -> Option<&'a GrammarElement> {
        grammar.get_rule(&self.rule).and_then(|rule| {
            rule.alternatives
                .get(self.alt_idx)
                .and_then(|alt| alt.elements.get(self.elem_idx))
        })
    }
}

/// Grammar state machine for tracking generation progress
#[derive(Debug, Clone)]
pub struct GrammarStateMachine {
    /// The grammar being enforced
    grammar: Grammar,
    /// Current possible states (NFA-style)
    states: Vec<GrammarState>,
    /// Generated characters so far
    generated: String,
}

impl GrammarStateMachine {
    /// Create new state machine from grammar
    ///
    /// # Errors
    ///
    /// Returns an error if the grammar fails validation.
    pub fn new(grammar: Grammar) -> Result<Self> {
        grammar.validate()?;

        let initial = GrammarState::initial(grammar.root());
        let mut states = vec![initial];

        // Expand initial states for all alternatives of root rule
        if let Some(root_rule) = grammar.get_rule(grammar.root()) {
            states.clear();
            for (alt_idx, _) in root_rule.alternatives.iter().enumerate() {
                states.push(GrammarState {
                    rule: grammar.root().to_string(),
                    alt_idx,
                    elem_idx: 0,
                    stack: Vec::new(),
                });
            }
        }

        Ok(Self {
            grammar,
            states,
            generated: String::new(),
        })
    }

    /// Check if a character is valid at current state
    pub fn is_valid_char(&self, c: char) -> bool {
        for state in &self.states {
            if self.can_accept_char(state, c) {
                return true;
            }
        }
        false
    }

    /// Get all valid characters at current state
    pub fn valid_chars(&self) -> HashSet<char> {
        let mut valid = HashSet::new();

        for state in &self.states {
            self.collect_valid_chars(state, &mut valid);
        }

        valid
    }

    /// Advance state machine with a character
    pub fn advance(&mut self, c: char) -> bool {
        let mut new_states = Vec::new();

        for state in &self.states {
            if let Some(next_states) = self.advance_state(state, c) {
                new_states.extend(next_states);
            }
        }

        if new_states.is_empty() {
            return false;
        }

        self.states = new_states;
        self.generated.push(c);
        true
    }

    /// Check if generation is complete (valid end state)
    pub fn is_complete(&self) -> bool {
        self.states.iter().any(|s| s.is_complete(&self.grammar))
    }

    /// Check if any valid continuation exists
    pub fn has_valid_continuation(&self) -> bool {
        !self.states.is_empty()
    }

    /// Get generated string so far
    pub fn generated(&self) -> &str {
        &self.generated
    }

    /// Reset state machine
    pub fn reset(&mut self) {
        let initial = GrammarState::initial(self.grammar.root());
        self.states = vec![initial];

        // Expand for all alternatives
        if let Some(root_rule) = self.grammar.get_rule(self.grammar.root()) {
            self.states.clear();
            for (alt_idx, _) in root_rule.alternatives.iter().enumerate() {
                self.states.push(GrammarState {
                    rule: self.grammar.root().to_string(),
                    alt_idx,
                    elem_idx: 0,
                    stack: Vec::new(),
                });
            }
        }

        self.generated.clear();
    }

    // Internal: Check if state can accept character
    fn can_accept_char(&self, state: &GrammarState, c: char) -> bool {
        let Some(elem) = state.current_element(&self.grammar) else {
            return false;
        };
        match elem {
            GrammarElement::Char(expected) => c == *expected,
            GrammarElement::CharRange(start, end) => c >= *start && c <= *end,
            GrammarElement::CharNot(excluded) => !excluded.contains(&c),
            GrammarElement::Any => true,
            GrammarElement::RuleRef(rule_name) => self.any_alternative_accepts(rule_name, c),
            GrammarElement::End => false,
        }
    }

    // Check if any alternative of the referenced rule accepts character c
    fn any_alternative_accepts(&self, rule_name: &str, c: char) -> bool {
        let Some(rule) = self.grammar.get_rule(rule_name) else {
            return false;
        };
        rule.alternatives.iter().enumerate().any(|(alt_idx, _)| {
            let sub_state = GrammarState {
                rule: rule_name.to_string(),
                alt_idx,
                elem_idx: 0,
                stack: Vec::new(),
            };
            self.can_accept_char(&sub_state, c)
        })
    }

    // Internal: Collect valid characters for a state
    fn collect_valid_chars(&self, state: &GrammarState, valid: &mut HashSet<char>) {
        let Some(elem) = state.current_element(&self.grammar) else {
            return;
        };
        match elem {
            GrammarElement::Char(c) => {
                valid.insert(*c);
            },
            GrammarElement::CharRange(start, end) => {
                for c in *start..=*end {
                    valid.insert(c);
                }
            },
            GrammarElement::CharNot(_) => {
                // For negated sets, check each printable char against the exclusion list
                for c in ' '..='~' {
                    if self.can_accept_char(state, c) {
                        valid.insert(c);
                    }
                }
            },
            GrammarElement::Any => {
                valid.extend(' '..='~');
            },
            GrammarElement::RuleRef(rule_name) => {
                self.collect_chars_from_alternatives(rule_name, valid);
            },
            GrammarElement::End => {},
        }
    }

    // Recurse into all alternatives of a referenced rule to collect valid chars
    fn collect_chars_from_alternatives(&self, rule_name: &str, valid: &mut HashSet<char>) {
        let Some(rule) = self.grammar.get_rule(rule_name) else {
            return;
        };
        for (alt_idx, _) in rule.alternatives.iter().enumerate() {
            let sub_state = GrammarState {
                rule: rule_name.to_string(),
                alt_idx,
                elem_idx: 0,
                stack: Vec::new(),
            };
            self.collect_valid_chars(&sub_state, valid);
        }
    }

    // Internal: Advance state and return new states
    fn advance_state(&self, state: &GrammarState, c: char) -> Option<Vec<GrammarState>> {
        let elem = state.current_element(&self.grammar)?;

        match elem {
            GrammarElement::Char(expected) => {
                if c == *expected {
                    Some(vec![self.next_state(state)])
                } else {
                    None
                }
            },
            GrammarElement::CharRange(start, end) => {
                if c >= *start && c <= *end {
                    Some(vec![self.next_state(state)])
                } else {
                    None
                }
            },
            GrammarElement::CharNot(excluded) => {
                if !excluded.contains(&c) {
                    Some(vec![self.next_state(state)])
                } else {
                    None
                }
            },
            GrammarElement::Any => Some(vec![self.next_state(state)]),
            GrammarElement::RuleRef(rule_name) => {
                // Enter referenced rule
                let rule = self.grammar.get_rule(rule_name)?;
                let mut new_states = Vec::new();

                for (alt_idx, _) in rule.alternatives.iter().enumerate() {
                    let mut sub_state = GrammarState {
                        rule: rule_name.clone(),
                        alt_idx,
                        elem_idx: 0,
                        stack: state.stack.clone(),
                    };
                    // Push return address
                    sub_state
                        .stack
                        .push((state.rule.clone(), state.alt_idx, state.elem_idx + 1));

                    if let Some(advanced) = self.advance_state(&sub_state, c) {
                        new_states.extend(advanced);
                    }
                }

                if new_states.is_empty() {
                    None
                } else {
                    Some(new_states)
                }
            },
            GrammarElement::End => None,
        }
    }

    // Internal: Create next state after consuming element
    fn next_state(&self, state: &GrammarState) -> GrammarState {
        let mut new_state = state.clone();
        new_state.elem_idx += 1;

        // Check if we've completed current alternative
        if let Some(rule) = self.grammar.get_rule(&state.rule) {
            if let Some(alt) = rule.alternatives.get(state.alt_idx) {
                if new_state.elem_idx >= alt.elements.len() {
                    // Pop from stack if there's a return address
                    if let Some((ret_rule, ret_alt, ret_elem)) = new_state.stack.pop() {
                        new_state.rule = ret_rule;
                        new_state.alt_idx = ret_alt;
                        new_state.elem_idx = ret_elem;
                    }
                }
            }
        }

        new_state
    }
}

// =============================================================================
// JSON SCHEMA GRAMMAR BUILDER
// =============================================================================

/// JSON Schema types for grammar generation
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum JsonSchemaType {
    /// String type
    String,
    /// Integer type
    Integer,
    /// Number type (float)
    Number,
    /// Boolean type
    Boolean,
    /// Null type
    Null,
    /// Array type with item schema
    Array(Box<JsonSchemaType>),
    /// Object type with properties
    Object(Vec<(String, JsonSchemaType, bool)>), // (name, type, required)
    /// Enum with allowed values
    Enum(Vec<String>),
    /// Any type
    Any,
}

/// Build grammar from JSON schema type
pub fn grammar_from_json_schema(schema: &JsonSchemaType) -> Grammar {
    let mut grammar = Grammar::with_root("root");

    // Add whitespace rule
    grammar.add_rule(GrammarRule::new(
        "ws",
        vec![
            GrammarAlternative::new(vec![]), // Empty (epsilon)
            GrammarAlternative::new(vec![
                GrammarElement::Char(' '),
                GrammarElement::RuleRef("ws".to_string()),
            ]),
            GrammarAlternative::new(vec![
                GrammarElement::Char('\n'),
                GrammarElement::RuleRef("ws".to_string()),
            ]),
            GrammarAlternative::new(vec![
                GrammarElement::Char('\t'),
                GrammarElement::RuleRef("ws".to_string()),
            ]),
        ],
    ));

    // Add digit rule
    grammar.add_rule(GrammarRule::new(
        "digit",
        vec![GrammarAlternative::new(vec![GrammarElement::CharRange(
            '0', '9',
        )])],
    ));

    // Add digits rule (one or more)
    grammar.add_rule(GrammarRule::new(
        "digits",
        vec![
            GrammarAlternative::new(vec![GrammarElement::RuleRef("digit".to_string())]),
            GrammarAlternative::new(vec![
                GrammarElement::RuleRef("digit".to_string()),
                GrammarElement::RuleRef("digits".to_string()),
            ]),
        ],
    ));

    // Add string character rule
    grammar.add_rule(GrammarRule::new(
        "string_char",
        vec![
            GrammarAlternative::new(vec![GrammarElement::CharNot(vec!['"', '\\', '\n'])]),
            GrammarAlternative::new(vec![GrammarElement::Char('\\'), GrammarElement::Char('"')]),
            GrammarAlternative::new(vec![GrammarElement::Char('\\'), GrammarElement::Char('\\')]),
            GrammarAlternative::new(vec![GrammarElement::Char('\\'), GrammarElement::Char('n')]),
        ],
    ));

    // Add string content rule
    grammar.add_rule(GrammarRule::new(
        "string_content",
        vec![
            GrammarAlternative::new(vec![]), // Empty
            GrammarAlternative::new(vec![
                GrammarElement::RuleRef("string_char".to_string()),
                GrammarElement::RuleRef("string_content".to_string()),
            ]),
        ],
    ));

    // Add base type rules based on schema
    add_schema_rules(&mut grammar, "root", schema);

    grammar
}

/// Add array schema rules
fn add_array_schema_rule(grammar: &mut Grammar, rule_name: &str, item_schema: &JsonSchemaType) {
    let item_rule = format!("{rule_name}_item");
    add_schema_rules(grammar, &item_rule, item_schema);

    let items_rule = format!("{rule_name}_items");
    grammar.add_rule(GrammarRule::new(
        &items_rule,
        vec![
            GrammarAlternative::new(vec![]),
            GrammarAlternative::new(vec![
                GrammarElement::Char(','),
                GrammarElement::RuleRef("ws".to_string()),
                GrammarElement::RuleRef(item_rule.clone()),
                GrammarElement::RuleRef(items_rule.clone()),
            ]),
        ],
    ));

    grammar.add_rule(GrammarRule::new(
        rule_name,
        vec![
            GrammarAlternative::new(vec![
                GrammarElement::Char('['),
                GrammarElement::RuleRef("ws".to_string()),
                GrammarElement::Char(']'),
            ]),
            GrammarAlternative::new(vec![
                GrammarElement::Char('['),
                GrammarElement::RuleRef("ws".to_string()),
                GrammarElement::RuleRef(item_rule),
                GrammarElement::RuleRef(items_rule),
                GrammarElement::RuleRef("ws".to_string()),
                GrammarElement::Char(']'),
            ]),
        ],
    ));
}

/// Add "any JSON value" schema rules
fn add_any_schema_rule(grammar: &mut Grammar, rule_name: &str) {
    grammar.add_rule(GrammarRule::new(
        rule_name,
        vec![
            GrammarAlternative::new(vec![GrammarElement::RuleRef("string_value".to_string())]),
            GrammarAlternative::new(vec![GrammarElement::RuleRef("number".to_string())]),
            GrammarAlternative::new(vec![GrammarElement::RuleRef("boolean".to_string())]),
            GrammarAlternative::new(vec![GrammarElement::RuleRef("null".to_string())]),
        ],
    ));

    if grammar.get_rule("string_value").is_none() {
        add_schema_rules(grammar, "string_value", &JsonSchemaType::String);
    }
    if grammar.get_rule("number").is_none() {
        add_schema_rules(grammar, "number", &JsonSchemaType::Number);
    }
    if grammar.get_rule("boolean").is_none() {
        add_schema_rules(grammar, "boolean", &JsonSchemaType::Boolean);
    }
    if grammar.get_rule("null").is_none() {
        add_schema_rules(grammar, "null", &JsonSchemaType::Null);
    }
}

fn add_schema_rules(grammar: &mut Grammar, rule_name: &str, schema: &JsonSchemaType) {
    match schema {
        JsonSchemaType::String => {
            grammar.add_rule(GrammarRule::new(
                rule_name,
                vec![GrammarAlternative::new(vec![
                    GrammarElement::Char('"'),
                    GrammarElement::RuleRef("string_content".to_string()),
                    GrammarElement::Char('"'),
                ])],
            ));
        },
        JsonSchemaType::Integer => {
            grammar.add_rule(GrammarRule::new(
                rule_name,
                vec![
                    GrammarAlternative::new(vec![GrammarElement::RuleRef("digits".to_string())]),
                    GrammarAlternative::new(vec![
                        GrammarElement::Char('-'),
                        GrammarElement::RuleRef("digits".to_string()),
                    ]),
                ],
            ));
        },
        JsonSchemaType::Number => {
            grammar.add_rule(GrammarRule::new(
                rule_name,
                vec![
                    GrammarAlternative::new(vec![GrammarElement::RuleRef("digits".to_string())]),
                    GrammarAlternative::new(vec![
                        GrammarElement::Char('-'),
                        GrammarElement::RuleRef("digits".to_string()),
                    ]),
                    GrammarAlternative::new(vec![
                        GrammarElement::RuleRef("digits".to_string()),
                        GrammarElement::Char('.'),
                        GrammarElement::RuleRef("digits".to_string()),
                    ]),
                    GrammarAlternative::new(vec![
                        GrammarElement::Char('-'),
                        GrammarElement::RuleRef("digits".to_string()),
                        GrammarElement::Char('.'),
                        GrammarElement::RuleRef("digits".to_string()),
                    ]),
                ],
            ));
        },
        JsonSchemaType::Boolean => {
            grammar.add_rule(GrammarRule::new(
                rule_name,
                vec![
                    GrammarAlternative::new(vec![
                        GrammarElement::Char('t'),
                        GrammarElement::Char('r'),
                        GrammarElement::Char('u'),
                        GrammarElement::Char('e'),
                    ]),
                    GrammarAlternative::new(vec![
                        GrammarElement::Char('f'),
                        GrammarElement::Char('a'),
                        GrammarElement::Char('l'),
                        GrammarElement::Char('s'),
                        GrammarElement::Char('e'),
                    ]),
                ],
            ));
        },
        JsonSchemaType::Null => {
            grammar.add_rule(GrammarRule::new(
                rule_name,
                vec![GrammarAlternative::new(vec![
                    GrammarElement::Char('n'),
                    GrammarElement::Char('u'),
                    GrammarElement::Char('l'),
                    GrammarElement::Char('l'),
                ])],
            ));
        },
        JsonSchemaType::Enum(values) => {
            let alternatives: Vec<GrammarAlternative> = values
                .iter()
                .map(|v| {
                    let mut elements = vec![GrammarElement::Char('"')];
                    for c in v.chars() {
                        elements.push(GrammarElement::Char(c));
                    }
                    elements.push(GrammarElement::Char('"'));
                    GrammarAlternative::new(elements)
                })
                .collect();
            grammar.add_rule(GrammarRule::new(rule_name, alternatives));
        },
        JsonSchemaType::Array(item_schema) => {
            add_array_schema_rule(grammar, rule_name, item_schema);
        },
        JsonSchemaType::Object(properties) => {
            add_object_schema_rules(grammar, rule_name, properties);
        },
        JsonSchemaType::Any => {
            add_any_schema_rule(grammar, rule_name);
        },
    }
}

/// Build grammar rules for a JSON object schema with the given properties
fn add_object_schema_rules(
    grammar: &mut Grammar,
    rule_name: &str,
    properties: &[(String, JsonSchemaType, bool)],
) {
    if properties.is_empty() {
        grammar.add_rule(GrammarRule::new(
            rule_name,
            vec![GrammarAlternative::new(vec![
                GrammarElement::Char('{'),
                GrammarElement::RuleRef("ws".to_string()),
                GrammarElement::Char('}'),
            ])],
        ));
        return;
    }

    let mut elements = vec![
        GrammarElement::Char('{'),
        GrammarElement::RuleRef("ws".to_string()),
    ];

    for (i, (prop_name, prop_type, _required)) in properties.iter().enumerate() {
        if i > 0 {
            elements.push(GrammarElement::Char(','));
            elements.push(GrammarElement::RuleRef("ws".to_string()));
        }

        // Property name
        elements.push(GrammarElement::Char('"'));
        for c in prop_name.chars() {
            elements.push(GrammarElement::Char(c));
        }
        elements.push(GrammarElement::Char('"'));
        elements.push(GrammarElement::RuleRef("ws".to_string()));
        elements.push(GrammarElement::Char(':'));
        elements.push(GrammarElement::RuleRef("ws".to_string()));

        // Property value
        let prop_rule = format!("{rule_name}_{prop_name}");
        add_schema_rules(grammar, &prop_rule, prop_type);
        elements.push(GrammarElement::RuleRef(prop_rule));
    }

    elements.push(GrammarElement::RuleRef("ws".to_string()));
    elements.push(GrammarElement::Char('}'));

    grammar.add_rule(GrammarRule::new(
        rule_name,
        vec![GrammarAlternative::new(elements)],
    ));
}

// =============================================================================
// TOKEN MASKING FOR CONSTRAINED GENERATION
// =============================================================================

/// Token mask for constrained generation
#[derive(Debug, Clone)]
pub struct TokenMask {
    /// Allowed token IDs
    pub allowed: HashSet<u32>,
    /// Whether to allow end-of-sequence
    pub allow_eos: bool,
}

impl TokenMask {
    /// Create mask allowing all tokens
    pub fn allow_all(vocab_size: usize) -> Self {
        Self {
            allowed: (0..vocab_size as u32).collect(),
            allow_eos: true,
        }
    }

    /// Create mask from allowed set
    pub fn from_allowed(allowed: HashSet<u32>, allow_eos: bool) -> Self {
        Self { allowed, allow_eos }
    }

    /// Check if token is allowed
    pub fn is_allowed(&self, token_id: u32) -> bool {
        self.allowed.contains(&token_id)
    }

    /// Apply mask to logits (set disallowed to -inf)
    pub fn apply_to_logits(&self, logits: &mut [f32]) {
        for (i, logit) in logits.iter_mut().enumerate() {
            if !self.allowed.contains(&(i as u32)) {
                *logit = f32::NEG_INFINITY;
            }
        }
    }

    /// Number of allowed tokens
    pub fn num_allowed(&self) -> usize {
        self.allowed.len()
    }
}

/// Grammar-based token masker
pub struct GrammarTokenMasker {
    /// State machine tracking grammar state
    state_machine: GrammarStateMachine,
    /// Token to string mapping
    token_strings: HashMap<u32, String>,
    /// EOS token ID
    eos_token_id: u32,
}

impl GrammarTokenMasker {
    /// Create new masker from grammar and vocabulary
    ///
    /// # Errors
    ///
    /// Returns an error if the grammar fails validation.
    pub fn new(
        grammar: Grammar,
        token_strings: HashMap<u32, String>,
        eos_token_id: u32,
    ) -> Result<Self> {
        let state_machine = GrammarStateMachine::new(grammar)?;
        Ok(Self {
            state_machine,
            token_strings,
            eos_token_id,
        })
    }

    /// Check if all characters in a multi-char token form a valid sequence
    fn is_token_valid_sequence(&self, token_str: &str) -> bool {
        let mut temp_sm = self.state_machine.clone();
        token_str.chars().all(|c| temp_sm.advance(c))
    }

    /// Get mask for current state
    pub fn get_mask(&self) -> TokenMask {
        let valid_chars = self.state_machine.valid_chars();
        let mut allowed = HashSet::new();

        for (token_id, token_str) in &self.token_strings {
            if let Some(first_char) = token_str.chars().next() {
                if valid_chars.contains(&first_char)
                    && (token_str.len() == 1 || self.is_token_valid_sequence(token_str))
                {
                    allowed.insert(*token_id);
                }
            }
        }

        TokenMask::from_allowed(allowed, self.state_machine.is_complete())
    }

    /// Advance masker with selected token
    pub fn advance_token(&mut self, token_id: u32) -> bool {
        if let Some(token_str) = self.token_strings.get(&token_id) {
            for c in token_str.chars() {
                if !self.state_machine.advance(c) {
                    return false;
                }
            }
            true
        } else {
            false
        }
    }

    /// Check if generation is complete
    pub fn is_complete(&self) -> bool {
        self.state_machine.is_complete()
    }

    /// Reset masker state
    pub fn reset(&mut self) {
        self.state_machine.reset();
    }

    /// Get EOS token ID
    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }
}

// =============================================================================
// TOOL CALLING / FUNCTION CALLING
// =============================================================================
//
// Implements OpenAI-style tool/function calling for LLM inference.
// Allows models to generate structured function calls that can be executed
// and results fed back into the conversation.
//
// Reference: OpenAI Function Calling API
// - Tool definitions with JSON Schema parameters
// - Tool choice: auto, required, none, or specific tool
// - Tool call parsing from model output
// - Grammar generation for constrained tool output

/// JSON Schema property type for tool parameters
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolParameterType {
    /// String parameter
    #[default]
    String,
    /// Integer parameter
    Integer,
    /// Number parameter (float)
    Number,
    /// Boolean parameter
    Boolean,
    /// Array parameter
    Array {
        /// Type of array items
        items: Box<ToolParameterType>,
    },
    /// Object parameter with properties
    Object {
        /// Properties of the object
        properties: Vec<ToolParameter>,
    },
    /// Enum of allowed string values
    Enum(Vec<String>),
}

/// Tool parameter definition
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolParameter {
    /// Parameter name
    pub name: String,
    /// Parameter description
    pub description: String,
    /// Parameter type
    #[serde(rename = "type")]
    pub param_type: ToolParameterType,
    /// Whether the parameter is required
    #[serde(default)]
    pub required: bool,
    /// Default value (JSON string)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<String>,
}

impl ToolParameter {
    /// Create new required string parameter
    pub fn required_string(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            param_type: ToolParameterType::String,
            required: true,
            default: None,
        }
    }

    /// Create new optional string parameter
    pub fn optional_string(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            param_type: ToolParameterType::String,
            required: false,
            default: None,
        }
    }

    /// Create new required integer parameter
    pub fn required_int(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            param_type: ToolParameterType::Integer,
            required: true,
            default: None,
        }
    }

    /// Create new enum parameter
    pub fn required_enum(
        name: impl Into<String>,
        description: impl Into<String>,
        values: Vec<String>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            param_type: ToolParameterType::Enum(values),
            required: true,
            default: None,
        }
    }

    /// Set default value
    #[must_use]
    pub fn with_default(mut self, default: impl Into<String>) -> Self {
        self.default = Some(default.into());
        self
    }
}

/// Tool/function definition
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Tool name (must be valid identifier: [a-zA-Z_][a-zA-Z0-9_]*)
    pub name: String,
    /// Tool description
    pub description: String,
    /// Tool parameters
    pub parameters: Vec<ToolParameter>,
}

impl ToolDefinition {
    /// Create new tool definition
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Vec<ToolParameter>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
        }
    }

    /// Get required parameters
    pub fn required_params(&self) -> impl Iterator<Item = &ToolParameter> {
        self.parameters.iter().filter(|p| p.required)
    }

    /// Get optional parameters
    pub fn optional_params(&self) -> impl Iterator<Item = &ToolParameter> {
        self.parameters.iter().filter(|p| !p.required)
    }

    /// Validate tool name is a valid identifier
    pub fn is_valid_name(name: &str) -> bool {
        if name.is_empty() {
            return false;
        }
        let mut chars = name.chars();
        // SAFETY: name is non-empty (checked above)
        let first = chars.next().expect("name is non-empty");
        if !first.is_ascii_alphabetic() && first != '_' {
            return false;
        }
        chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
    }
}

/// Tool choice mode
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoice {
    /// Model decides whether to call a tool
    #[default]
    Auto,
    /// Model must call at least one tool
    Required,
    /// Model must not call any tools
    None,
    /// Model must call the specified tool
    Specific(String),
}

/// Parsed tool call from model output
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique ID for this tool call
    pub id: String,
    /// Tool name
    pub name: String,
    /// Tool arguments as JSON string
    pub arguments: String,
}

impl ToolCall {
    /// Create new tool call
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            arguments: arguments.into(),
        }
    }

    /// Parse arguments as JSON value
    ///
    /// # Errors
    ///
    /// Returns `RealizarError::InvalidConfiguration` if the arguments are not valid JSON.
    pub fn parse_arguments(&self) -> Result<serde_json::Value> {
        serde_json::from_str(&self.arguments).map_err(|e| {
            RealizarError::InvalidConfiguration(format!("Failed to parse tool arguments: {e}"))
        })
    }
}

/// Tool call result
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolResult {
    /// Tool call ID this result is for
    pub tool_call_id: String,
    /// Result content (JSON or plain text)
    pub content: String,
    /// Whether the tool call succeeded
    #[serde(default = "default_true")]
    pub success: bool,
}

fn default_true() -> bool {
    true
}

impl ToolResult {
    /// Create successful result
    pub fn success(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            content: content.into(),
            success: true,
        }
    }

    /// Create error result
    pub fn error(tool_call_id: impl Into<String>, error: impl Into<String>) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            content: error.into(),
            success: false,
        }
    }
}

/// Tool call parser for extracting tool calls from model output
pub struct ToolCallParser {
    /// Available tools
    tools: Vec<ToolDefinition>,
    /// Tool call format (default: OpenAI-style JSON)
    format: ToolCallFormat,
    /// Next tool call ID
    next_id: u64,
}

/// Format for tool calls in model output
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ToolCallFormat {
    /// OpenAI-style JSON: {"name": "tool", "arguments": {...}}
    #[default]
    OpenAI,
    /// Anthropic-style XML: <tool_use><name>tool</name><input>{...}</input></tool_use>
    Anthropic,
    /// Hermes format: <tool_call>{"name": "tool", "arguments": {...}}</tool_call>
    Hermes,
}

impl ToolCallParser {
    /// Create new parser with tools
    pub fn new(tools: Vec<ToolDefinition>) -> Self {
        Self {
            tools,
            format: ToolCallFormat::default(),
            next_id: 0,
        }
    }

    /// Set tool call format
    #[must_use]
    pub fn with_format(mut self, format: ToolCallFormat) -> Self {
        self.format = format;
        self
    }

    /// Generate unique tool call ID
    pub fn generate_id(&mut self) -> String {
        let id = format!("call_{}", self.next_id);
        self.next_id += 1;
        id
    }

    /// Get available tool names
    pub fn tool_names(&self) -> impl Iterator<Item = &str> {
        self.tools.iter().map(|t| t.name.as_str())
    }

    /// Get tool by name
    pub fn get_tool(&self, name: &str) -> Option<&ToolDefinition> {
        self.tools.iter().find(|t| t.name == name)
    }

    /// Parse tool calls from text (OpenAI format)
    ///
    /// Looks for JSON objects with "name" and "arguments" fields.
    pub fn parse(&mut self, text: &str) -> Vec<ToolCall> {
        match self.format {
            ToolCallFormat::OpenAI => self.parse_openai(text),
            ToolCallFormat::Anthropic => self.parse_anthropic(text),
            ToolCallFormat::Hermes => self.parse_hermes(text),
        }
    }

    /// Try to extract a tool call from a parsed JSON value with "name" and "arguments" fields.
    fn try_extract_json_tool_call(&mut self, value: &serde_json::Value) -> Option<ToolCall> {
        let name = value.get("name").and_then(|v| v.as_str())?;
        let args = value.get("arguments")?;
        if !self.tools.iter().any(|t| t.name == name) {
            return None;
        }
        let arguments = if args.is_string() {
            args.as_str().expect("checked is_string above").to_string()
        } else {
            args.to_string()
        };
        Some(ToolCall::new(self.generate_id(), name, arguments))
    }

    fn parse_openai(&mut self, text: &str) -> Vec<ToolCall> {
        let mut calls = Vec::new();

        // Try to find JSON objects with tool call structure
        // Look for patterns like {"name": "...", "arguments": {...}}
        let mut start = 0;
        while let Some(pos) = text[start..].find('{') {
            let abs_pos = start + pos;
            if let Some(end) = find_matching_brace(&text[abs_pos..]) {
                let json_str = &text[abs_pos..=(abs_pos + end)];
                if let Ok(value) = serde_json::from_str::<serde_json::Value>(json_str) {
                    if let Some(call) = self.try_extract_json_tool_call(&value) {
                        calls.push(call);
                    }
                }
                start = abs_pos + end + 1;
            } else {
                start = abs_pos + 1;
            }
        }

        calls
    }

    fn parse_anthropic(&mut self, text: &str) -> Vec<ToolCall> {
        let mut calls = Vec::new();

        // Look for <tool_use>...</tool_use> blocks
        let mut pos = 0;
        while let Some(start) = text[pos..].find("<tool_use>") {
            let abs_start = pos + start + 10; // Skip "<tool_use>"
            if let Some(end) = text[abs_start..].find("</tool_use>") {
                let content = &text[abs_start..abs_start + end];

                // Extract name
                let name = extract_xml_tag(content, "name");
                let input = extract_xml_tag(content, "input");

                if let (Some(name), Some(input)) = (name, input) {
                    if self.tools.iter().any(|t| t.name == name) {
                        calls.push(ToolCall::new(self.generate_id(), name, input));
                    }
                }

                pos = abs_start + end + 11; // Skip "</tool_use>"
            } else {
                break;
            }
        }

        calls
    }

    fn parse_hermes(&mut self, text: &str) -> Vec<ToolCall> {
        let mut calls = Vec::new();

        // Look for <tool_call>...</tool_call> blocks
        let mut pos = 0;
        while let Some(start) = text[pos..].find("<tool_call>") {
            let abs_start = pos + start + 11; // Skip "<tool_call>"
            if let Some(end) = text[abs_start..].find("</tool_call>") {
                let json_str = text[abs_start..abs_start + end].trim();

                if let Ok(value) = serde_json::from_str::<serde_json::Value>(json_str) {
                    if let Some(call) = self.try_extract_json_tool_call(&value) {
                        calls.push(call);
                    }
                }

                pos = abs_start + end + 12; // Skip "</tool_call>"
            } else {
                break;
            }
        }

        calls
    }
}

/// Process a single character for brace matching state machine.
/// Returns (depth_delta, toggle_string, set_escape) for the char.
#[inline]
fn process_brace_char(c: char, in_string: bool) -> (i32, bool, bool) {
    match c {
        '\\' if in_string => (0, false, true),
        '"' => (0, true, false),
        '{' if !in_string => (1, false, false),
        '}' if !in_string => (-1, false, false),
        _ => (0, false, false),
    }
}

/// Find matching closing brace, handling nested braces
fn find_matching_brace(s: &str) -> Option<usize> {
    let mut depth = 0;
    let mut in_string = false;
    let mut escape_next = false;

    for (i, c) in s.char_indices() {
        if escape_next {
            escape_next = false;
            continue;
        }

        let (delta, toggle, escape) = process_brace_char(c, in_string);
        depth += delta;
        if toggle {
            in_string = !in_string;
        }
        escape_next = escape;

        if depth == 0 && delta < 0 {
            return Some(i);
        }
    }

    None
}

/// Extract content of an XML tag
fn extract_xml_tag(content: &str, tag: &str) -> Option<String> {
    let open_tag = format!("<{tag}>");
    let close_tag = format!("</{tag}>");

    if let Some(start) = content.find(&open_tag) {
        let value_start = start + open_tag.len();
        if let Some(end) = content[value_start..].find(&close_tag) {
            return Some(content[value_start..value_start + end].to_string());
        }
    }
    None
}

/// Generate grammar for tool calling output
///
/// Creates a grammar that constrains model output to valid tool calls.
pub fn generate_tool_grammar(tools: &[ToolDefinition]) -> Grammar {
    let mut grammar = Grammar::default();

    // Add basic rules
    add_json_whitespace_rules(&mut grammar);

    // Generate alternatives for each tool
    let mut tool_alternatives = Vec::new();

    for tool in tools {
        let tool_rule = format!("tool_{}", tool.name);

        // Generate parameter object grammar
        let params_rule = format!("{tool_rule}_params");
        generate_params_grammar(&mut grammar, &params_rule, &tool.parameters);

        // Tool call rule: {"name": "tool_name", "arguments": {...}}
        let mut elements = vec![
            GrammarElement::Char('{'),
            GrammarElement::RuleRef("ws".to_string()),
            GrammarElement::Char('"'),
            GrammarElement::Char('n'),
            GrammarElement::Char('a'),
            GrammarElement::Char('m'),
            GrammarElement::Char('e'),
            GrammarElement::Char('"'),
            GrammarElement::RuleRef("ws".to_string()),
            GrammarElement::Char(':'),
            GrammarElement::RuleRef("ws".to_string()),
            GrammarElement::Char('"'),
        ];

        // Tool name literal
        for c in tool.name.chars() {
            elements.push(GrammarElement::Char(c));
        }

        elements.extend([
            GrammarElement::Char('"'),
            GrammarElement::RuleRef("ws".to_string()),
            GrammarElement::Char(','),
            GrammarElement::RuleRef("ws".to_string()),
            GrammarElement::Char('"'),
            GrammarElement::Char('a'),
            GrammarElement::Char('r'),
            GrammarElement::Char('g'),
            GrammarElement::Char('u'),
            GrammarElement::Char('m'),
            GrammarElement::Char('e'),
            GrammarElement::Char('n'),
            GrammarElement::Char('t'),
            GrammarElement::Char('s'),
            GrammarElement::Char('"'),
            GrammarElement::RuleRef("ws".to_string()),
            GrammarElement::Char(':'),
            GrammarElement::RuleRef("ws".to_string()),
            GrammarElement::RuleRef(params_rule),
            GrammarElement::RuleRef("ws".to_string()),
            GrammarElement::Char('}'),
        ]);

        grammar.add_rule(GrammarRule::new(
            &tool_rule,
            vec![GrammarAlternative::new(elements)],
        ));

        tool_alternatives.push(GrammarAlternative::new(vec![GrammarElement::RuleRef(
            tool_rule,
        )]));
    }

    // Root rule: one of the tools
    grammar.add_rule(GrammarRule::new("root", tool_alternatives));

    grammar
}

/// Generate grammar for tool parameters
fn generate_params_grammar(grammar: &mut Grammar, rule_name: &str, params: &[ToolParameter]) {
    if params.is_empty() {
        // Empty object: {}
        grammar.add_rule(GrammarRule::new(
            rule_name,
            vec![GrammarAlternative::new(vec![
                GrammarElement::Char('{'),
                GrammarElement::RuleRef("ws".to_string()),
                GrammarElement::Char('}'),
            ])],
        ));
        return;
    }

    // Build object with all parameters
    let mut elements = vec![
        GrammarElement::Char('{'),
        GrammarElement::RuleRef("ws".to_string()),
    ];

    for (i, param) in params.iter().enumerate() {
        if i > 0 {
            elements.push(GrammarElement::Char(','));
            elements.push(GrammarElement::RuleRef("ws".to_string()));
        }

        // Property name
        elements.push(GrammarElement::Char('"'));
        for c in param.name.chars() {
            elements.push(GrammarElement::Char(c));
        }
        elements.push(GrammarElement::Char('"'));
        elements.push(GrammarElement::RuleRef("ws".to_string()));
        elements.push(GrammarElement::Char(':'));
        elements.push(GrammarElement::RuleRef("ws".to_string()));

        // Property value based on type
        let value_rule = format!("{rule_name}_{}", param.name);
        generate_param_type_grammar(grammar, &value_rule, &param.param_type);
        elements.push(GrammarElement::RuleRef(value_rule));
    }

    elements.push(GrammarElement::RuleRef("ws".to_string()));
    elements.push(GrammarElement::Char('}'));

    grammar.add_rule(GrammarRule::new(
        rule_name,
        vec![GrammarAlternative::new(elements)],
    ));
}

/// Generate grammar for a parameter type
fn generate_param_type_grammar(
    grammar: &mut Grammar,
    rule_name: &str,
    param_type: &ToolParameterType,
) {
    match param_type {
        ToolParameterType::String => {
            grammar.add_rule(GrammarRule::new(
                rule_name,
                vec![GrammarAlternative::new(vec![GrammarElement::RuleRef(
                    "string".to_string(),
                )])],
            ));
        },
        ToolParameterType::Integer => {
            grammar.add_rule(GrammarRule::new(
                rule_name,
                vec![GrammarAlternative::new(vec![GrammarElement::RuleRef(
                    "integer".to_string(),
                )])],
            ));
        },
        ToolParameterType::Number => {
            grammar.add_rule(GrammarRule::new(
                rule_name,
                vec![GrammarAlternative::new(vec![GrammarElement::RuleRef(
                    "number".to_string(),
                )])],
            ));
        },
        ToolParameterType::Boolean => {
            grammar.add_rule(GrammarRule::new(
                rule_name,
                vec![GrammarAlternative::new(vec![GrammarElement::RuleRef(
                    "boolean".to_string(),
                )])],
            ));
        },
        ToolParameterType::Enum(values) => {
            let alternatives: Vec<_> = values
                .iter()
                .map(|v| {
                    let mut chars = vec![GrammarElement::Char('"')];
                    chars.extend(v.chars().map(GrammarElement::Char));
                    chars.push(GrammarElement::Char('"'));
                    GrammarAlternative::new(chars)
                })
                .collect();
            grammar.add_rule(GrammarRule::new(rule_name, alternatives));
        },
        ToolParameterType::Array { items } => {
            let item_rule = format!("{rule_name}_item");
            generate_param_type_grammar(grammar, &item_rule, items);

            let items_rule = format!("{rule_name}_items");
            grammar.add_rule(GrammarRule::new(
                &items_rule,
                vec![
                    GrammarAlternative::new(vec![]), // Empty
                    GrammarAlternative::new(vec![
                        GrammarElement::Char(','),
                        GrammarElement::RuleRef("ws".to_string()),
                        GrammarElement::RuleRef(item_rule.clone()),
                        GrammarElement::RuleRef(items_rule.clone()),
                    ]),
                ],
            ));

            grammar.add_rule(GrammarRule::new(
                rule_name,
                vec![
                    GrammarAlternative::new(vec![
                        GrammarElement::Char('['),
                        GrammarElement::RuleRef("ws".to_string()),
                        GrammarElement::Char(']'),
                    ]),
                    GrammarAlternative::new(vec![
                        GrammarElement::Char('['),
                        GrammarElement::RuleRef("ws".to_string()),
                        GrammarElement::RuleRef(item_rule),
                        GrammarElement::RuleRef(items_rule),
                        GrammarElement::RuleRef("ws".to_string()),
                        GrammarElement::Char(']'),
                    ]),
                ],
            ));
        },
        ToolParameterType::Object { properties } => {
            generate_params_grammar(grammar, rule_name, properties);
        },
    }
}

/// Add standard JSON whitespace rules to grammar
fn add_json_whitespace_rules(grammar: &mut Grammar) {
    // Whitespace (optional)
    grammar.add_rule(GrammarRule::new(
        "ws",
        vec![
            GrammarAlternative::new(vec![]),
            GrammarAlternative::new(vec![
                GrammarElement::Char(' '),
                GrammarElement::RuleRef("ws".to_string()),
            ]),
            GrammarAlternative::new(vec![
                GrammarElement::Char('\n'),
                GrammarElement::RuleRef("ws".to_string()),
            ]),
            GrammarAlternative::new(vec![
                GrammarElement::Char('\t'),
                GrammarElement::RuleRef("ws".to_string()),
            ]),
        ],
    ));

    // String (simplified - no escape handling)
    grammar.add_rule(GrammarRule::new(
        "string",
        vec![GrammarAlternative::new(vec![
            GrammarElement::Char('"'),
            GrammarElement::RuleRef("string_chars".to_string()),
            GrammarElement::Char('"'),
        ])],
    ));

    grammar.add_rule(GrammarRule::new(
        "string_chars",
        vec![
            GrammarAlternative::new(vec![]),
            GrammarAlternative::new(vec![
                GrammarElement::CharNot(vec!['"', '\\']),
                GrammarElement::RuleRef("string_chars".to_string()),
            ]),
        ],
    ));

    // Integer
    grammar.add_rule(GrammarRule::new(
        "integer",
        vec![
            GrammarAlternative::new(vec![
                GrammarElement::Char('-'),
                GrammarElement::RuleRef("digits".to_string()),
            ]),
            GrammarAlternative::new(vec![GrammarElement::RuleRef("digits".to_string())]),
        ],
    ));

    // Number (with optional decimal)
    grammar.add_rule(GrammarRule::new(
        "number",
        vec![
            GrammarAlternative::new(vec![
                GrammarElement::RuleRef("integer".to_string()),
                GrammarElement::Char('.'),
                GrammarElement::RuleRef("digits".to_string()),
            ]),
            GrammarAlternative::new(vec![GrammarElement::RuleRef("integer".to_string())]),
        ],
    ));

    // Digits
    grammar.add_rule(GrammarRule::new(
        "digits",
        vec![
            GrammarAlternative::new(vec![
                GrammarElement::CharRange('0', '9'),
                GrammarElement::RuleRef("digits".to_string()),
            ]),
            GrammarAlternative::new(vec![GrammarElement::CharRange('0', '9')]),
        ],
    ));

    // Boolean
    grammar.add_rule(GrammarRule::new(
        "boolean",
        vec![
            GrammarAlternative::new(vec![
                GrammarElement::Char('t'),
                GrammarElement::Char('r'),
                GrammarElement::Char('u'),
                GrammarElement::Char('e'),
            ]),
            GrammarAlternative::new(vec![
                GrammarElement::Char('f'),
                GrammarElement::Char('a'),
                GrammarElement::Char('l'),
                GrammarElement::Char('s'),
                GrammarElement::Char('e'),
            ]),
        ],
    ));
}

// =============================================================================
// TESTS
// =============================================================================

// Tests extracted to tests.rs (PMAT-802)
#[cfg(test)]
#[path = "tests.rs"]
mod grammar_tests;
