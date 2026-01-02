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
        if let Some(elem) = state.current_element(&self.grammar) {
            match elem {
                GrammarElement::Char(expected) => c == *expected,
                GrammarElement::CharRange(start, end) => c >= *start && c <= *end,
                GrammarElement::CharNot(excluded) => !excluded.contains(&c),
                GrammarElement::Any => true,
                GrammarElement::RuleRef(rule_name) => {
                    // Need to check if any alternative of referenced rule accepts c
                    if let Some(rule) = self.grammar.get_rule(rule_name) {
                        for (alt_idx, _) in rule.alternatives.iter().enumerate() {
                            let sub_state = GrammarState {
                                rule: rule_name.clone(),
                                alt_idx,
                                elem_idx: 0,
                                stack: Vec::new(),
                            };
                            if self.can_accept_char(&sub_state, c) {
                                return true;
                            }
                        }
                    }
                    false
                },
                GrammarElement::End => false,
            }
        } else {
            false
        }
    }

    // Internal: Collect valid characters for a state
    fn collect_valid_chars(&self, state: &GrammarState, valid: &mut HashSet<char>) {
        if let Some(elem) = state.current_element(&self.grammar) {
            match elem {
                GrammarElement::Char(c) => {
                    valid.insert(*c);
                },
                GrammarElement::CharRange(start, end) => {
                    for c in *start..=*end {
                        valid.insert(c);
                    }
                },
                GrammarElement::CharNot(_excluded) => {
                    // For negated sets, we'd need to add all chars except excluded
                    // This is expensive, so for now we mark as "any printable"
                    for c in ' '..='~' {
                        if self.can_accept_char(state, c) {
                            valid.insert(c);
                        }
                    }
                },
                GrammarElement::Any => {
                    // Add common printable characters
                    for c in ' '..='~' {
                        valid.insert(c);
                    }
                },
                GrammarElement::RuleRef(rule_name) => {
                    // Recurse into referenced rule
                    if let Some(rule) = self.grammar.get_rule(rule_name) {
                        for (alt_idx, _) in rule.alternatives.iter().enumerate() {
                            let sub_state = GrammarState {
                                rule: rule_name.clone(),
                                alt_idx,
                                elem_idx: 0,
                                stack: Vec::new(),
                            };
                            self.collect_valid_chars(&sub_state, valid);
                        }
                    }
                },
                GrammarElement::End => {},
            }
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
            // integer or decimal
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
            let item_rule = format!("{rule_name}_item");
            add_schema_rules(grammar, &item_rule, item_schema);

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
                    // Empty array
                    GrammarAlternative::new(vec![
                        GrammarElement::Char('['),
                        GrammarElement::RuleRef("ws".to_string()),
                        GrammarElement::Char(']'),
                    ]),
                    // Non-empty array
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
        JsonSchemaType::Object(properties) => {
            // Build object grammar with properties
            if properties.is_empty() {
                // Empty object
                grammar.add_rule(GrammarRule::new(
                    rule_name,
                    vec![GrammarAlternative::new(vec![
                        GrammarElement::Char('{'),
                        GrammarElement::RuleRef("ws".to_string()),
                        GrammarElement::Char('}'),
                    ])],
                ));
            } else {
                // Object with properties
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
        },
        JsonSchemaType::Any => {
            // Any JSON value
            grammar.add_rule(GrammarRule::new(
                rule_name,
                vec![
                    GrammarAlternative::new(vec![GrammarElement::RuleRef(
                        "string_value".to_string(),
                    )]),
                    GrammarAlternative::new(vec![GrammarElement::RuleRef("number".to_string())]),
                    GrammarAlternative::new(vec![GrammarElement::RuleRef("boolean".to_string())]),
                    GrammarAlternative::new(vec![GrammarElement::RuleRef("null".to_string())]),
                ],
            ));

            // Add helper rules if not present
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
        },
    }
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

    /// Get mask for current state
    pub fn get_mask(&self) -> TokenMask {
        let valid_chars = self.state_machine.valid_chars();
        let mut allowed = HashSet::new();

        for (token_id, token_str) in &self.token_strings {
            // Check if token's first character is valid
            if let Some(first_char) = token_str.chars().next() {
                if valid_chars.contains(&first_char) {
                    // For single-char tokens, this is sufficient
                    // For multi-char tokens, we'd need to simulate all chars
                    if token_str.len() == 1 {
                        allowed.insert(*token_id);
                    } else {
                        // Check if all characters in token are valid sequence
                        let mut temp_sm = self.state_machine.clone();
                        let mut all_valid = true;
                        for c in token_str.chars() {
                            if !temp_sm.advance(c) {
                                all_valid = false;
                                break;
                            }
                        }
                        if all_valid {
                            allowed.insert(*token_id);
                        }
                    }
                }
            }
        }

        let allow_eos = self.state_machine.is_complete();

        TokenMask::from_allowed(allowed, allow_eos)
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
                    if let (Some(name), Some(args)) = (
                        value.get("name").and_then(|v| v.as_str()),
                        value.get("arguments"),
                    ) {
                        // Check if this is a valid tool
                        if self.tools.iter().any(|t| t.name == name) {
                            let arguments = if args.is_string() {
                                args.as_str().expect("checked is_string above").to_string()
                            } else {
                                args.to_string()
                            };
                            calls.push(ToolCall::new(self.generate_id(), name, arguments));
                        }
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
                    if let (Some(name), Some(args)) = (
                        value.get("name").and_then(|v| v.as_str()),
                        value.get("arguments"),
                    ) {
                        if self.tools.iter().any(|t| t.name == name) {
                            let arguments = if args.is_string() {
                                args.as_str().expect("checked is_string above").to_string()
                            } else {
                                args.to_string()
                            };
                            calls.push(ToolCall::new(self.generate_id(), name, arguments));
                        }
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

        match c {
            '\\' if in_string => escape_next = true,
            '"' => in_string = !in_string,
            '{' if !in_string => depth += 1,
            '}' if !in_string => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            },
            _ => {},
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
