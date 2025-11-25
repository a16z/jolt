# Understanding Transpilation: From Theory to Jolt-Gnark Conversion

**Purpose**: Learn transpilation from first principles, then apply to converting Jolt verifier to Gnark circuits

**Audience**: Someone new to computer science and transpilation

**Structure**:
1. **Part I: Theory** - What is transpilation? How does it work?
2. **Part II: Toy Example** - Build a simple transpiler step-by-step
3. **Part III: Jolt Application** - Apply to real-world Jolt→Gnark conversion

**Date**: 2025-11-10

---

# Part I: Transpilation Theory

## 1. What is Transpilation?

### 1.1 The Translation Analogy

Imagine you have a recipe written in English, and you need it in Spanish. You have two options:

1. **Manual translation**: Read the English recipe and carefully rewrite it in Spanish by hand
2. **Automatic translation**: Use Google Translate to convert it automatically

**Transpilation** is like automatic translation, but for programming languages.

### 1.2 Formal Definition

**Transpilation** (source-to-source compilation) = Automatically converting code from one high-level programming language to another high-level programming language, while preserving the same logic.

**Example**: TypeScript → JavaScript
```typescript
// TypeScript (source)
function add(x: number, y: number): number {
    return x + y;
}
```
↓ *transpile* ↓
```javascript
// JavaScript (target)
function add(x, y) {
    return x + y;
}
```

Same logic, different syntax.

### 1.3 Transpilation vs Compilation

| Process | Input | Output | Example |
|---------|-------|--------|---------|
| **Compilation** | High-level code | Low-level code (machine code) | C → x86 assembly |
| **Transpilation** | High-level code | High-level code | TypeScript → JavaScript |

Both involve understanding source code and generating target code, but transpilation keeps things at a similar abstraction level.

### 1.4 Why Transpilation Matters

**Benefits**:
- ✅ **Single source of truth**: Write code once, generate multiple versions
- ✅ **Less error-prone**: Machines don't make typos or forget to update things
- ✅ **Maintainable**: When the original changes, re-run the transpiler to update the output
- ✅ **Multi-target**: One source can target multiple languages

**Costs**:
- ❌ **Upfront work**: Building a good transpiler takes effort
- ❌ **IR design**: Need to design intermediate representation carefully
- ❌ **Debugging**: Harder to debug generated code

---

## 2. The Transpilation Pipeline

Every transpiler follows the same basic pattern:

```
Source Code → Parse → IR → Generate → Target Code
```

Let's break down each step:

### 2.1 Parsing (Understanding the Source)

**Input**: Raw source code (text file)
```rust
fn verify(x: u32, y: u32) -> bool {
    x + y == 42
}
```

**Process**: Break into structured pieces (Abstract Syntax Tree - AST)
```
FunctionDeclaration
├── name: "verify"
├── parameters: [x: u32, y: u32]
├── return_type: bool
└── body:
    └── BinaryExpression
        ├── left: BinaryExpression (x + y)
        ├── operator: "=="
        └── right: Literal(42)
```

**Analogy**: Like diagramming a sentence in English class - identify subject, verb, object.

**Tools**: Parser generators (e.g., Rust's `syn` crate) or hand-written parsers

### 2.2 Analysis (Understanding Semantics)

**Input**: AST from parsing

**Process**: Figure out *what* the code does, not just *how* it's written
- Type checking: "Is `x + y` valid? Both are `u32`, so yes"
- Control flow: "What order do operations happen?"
- Dependencies: "What other code does this need?"

**Output**: Annotated AST with semantic information

### 2.3 Transformation (IR Creation)

**Input**: Annotated AST

**Process**: Convert to a **neutral format** (Intermediate Representation) that's easier to work with
- Remove language-specific quirks
- Normalize patterns
- Make implicit things explicit

**Output**: Intermediate Representation (IR)
```
IR {
    operations: [
        Op1: Add(Var(x), Var(y)) -> Temp1
        Op2: Equals(Temp1, Const(42)) -> Result
    ]
}
```

**Why IR?**
- Easier to analyze than raw AST
- Can optimize here (constant folding, dead code elimination)
- Can target multiple backends from one IR
- Language-neutral (not specific to source or target language)

### 2.4 Code Generation (Creating Target Code)

**Input**: IR

**Process**: Generate code in target language that implements the IR
- Map IR operations to target language constructs
- Add appropriate syntax (semicolons, braces, etc.)
- Format/indent nicely

**Output**: Target code
```go
// Go version
func verify(x uint32, y uint32) bool {
    return x + y == 42
}
```

---

## 3. The Intermediate Representation (IR)

The IR is the **most important** part of a transpiler. It's the bridge between languages.

### 3.1 What is an IR?

**Definition**: A language-neutral data structure that represents programs

**Key insight**: The IR is NOT specific to each program - it defines a **mini programming language** that can represent MANY programs.

**Analogy**:
- IR = Grammar rules of a language
- Each program = A sentence in that language
- You don't change grammar to say different sentences!
- You only change grammar to support new kinds of sentences

### 3.2 IR Scope

The IR defines what programs you CAN transpile:
- Small IR scope → Easy to implement, limited programs
- Large IR scope → Hard to implement, more programs

**Design strategy**:
1. Start small
2. Test with simple programs
3. Add features incrementally
4. Each addition requires updating parser AND generator

### 3.3 When to Modify the IR

**You modify IR when**: You want to support NEW KINDS of operations

**You DON'T modify IR when**: You transpile a different program

Example:
- Transpile `add(x, y)` → IR stays same ❌
- Transpile `subtract(x, y)` but IR doesn't support subtraction → Modify IR ✅

---

## 4. Can We Use Existing Transpilers?

For our use case (Rust → Go for cryptographic circuits), existing transpilers don't work well.

**Why?**
- Most transpilers target similar languages (TypeScript→JavaScript, C→C++)
- Rust→Go is very different:
  - Memory model: Ownership/borrowing vs garbage collection
  - Error handling: Result<T,E> vs error return values
  - Generics: Different syntax and semantics

**Existing tools**:
- `corrode` (C→Rust): Wrong direction
- `c2go` (C→Go): Closer, but we're starting from Rust
- LLVM IR→Go: Loses high-level structure

**Our approach**: Build a **custom extractor** that understands our specific domain
- We only care about mathematical operations (field arithmetic, polynomial evaluation)
- Don't need to handle all of Rust (no need for traits, lifetimes, etc.)
- Can optimize for our specific use case

---

# Interlude: Understanding ASTs and Parsers

Before we build our transpiler, let's understand the fundamental tools: **Abstract Syntax Trees (ASTs)** and **parsers**.

## 4a. What is a Parser?

### The Problem: How Do Computers Read Code?

To you, this makes sense:
```rust
fn add(x: i32, y: i32) -> i32 {
    x + y
}
```

But to a computer, it's just a string of characters:
```
"fn add(x: i32, y: i32) -> i32 {\n    x + y\n}"
```

**A parser** converts this string into a data structure that the computer can understand and work with.

### Analogy: Reading a Sentence

When you read the sentence "The cat sat on the mat", your brain automatically understands:
- **Subject**: "The cat"
- **Verb**: "sat"
- **Prepositional phrase**: "on the mat"

A parser does the same thing for code - it breaks it into meaningful pieces.

### What Parsers Do

**Input**: Text (string of characters)
```rust
"fn add(x: i32, y: i32) -> i32 { x + y }"
```

**Process**: Recognize patterns and structure
- "fn" → This is a function declaration
- "add" → This is the function name
- "(x: i32, y: i32)" → These are parameters
- "-> i32" → This is the return type
- "{ x + y }" → This is the function body

**Output**: Structured data (Abstract Syntax Tree)

---

## 4b. What is an Abstract Syntax Tree (AST)?

### Definition

An **Abstract Syntax Tree** is a tree-shaped data structure that represents the structure of code.

- **Tree**: Like a family tree, with one root and many branches
- **Abstract**: Ignores formatting details (spaces, newlines) and focuses on meaning
- **Syntax**: Represents the grammatical structure of the code

### Visual Example

**Code**:
```rust
fn add(x: i32, y: i32) -> i32 {
    x + y
}
```

**AST** (simplified):
```
FunctionDeclaration
├── name: "add"
├── parameters
│   ├── Parameter
│   │   ├── name: "x"
│   │   └── type: "i32"
│   └── Parameter
│       ├── name: "y"
│       └── type: "i32"
├── return_type: "i32"
└── body
    └── BinaryExpression
        ├── operator: "+"
        ├── left: Identifier("x")
        └── right: Identifier("y")
```

### Why Trees?

**Code is hierarchical**:
- A function contains parameters
- Parameters contain names and types
- The body contains expressions
- Expressions contain sub-expressions

Trees naturally represent this hierarchy!

### Concrete Example: Mathematical Expression

**Code**: `(2 + 3) * 4`

**As a string**: Just characters: `"(2 + 3) * 4"`

**As an AST**:
```
       Multiply(*)
      /           \
   Add(+)          4
   /    \
  2      3
```

**Why this matters**: The tree shows that we must add 2+3 BEFORE multiplying by 4. The structure encodes the order of operations!

### Another Example: If Statement

**Code**:
```rust
if x > 5 {
    print("big");
} else {
    print("small");
}
```

**AST**:
```
IfStatement
├── condition
│   └── BinaryExpression
│       ├── operator: ">"
│       ├── left: Identifier("x")
│       └── right: Literal(5)
├── then_branch
│   └── FunctionCall
│       ├── function: "print"
│       └── arguments: [Literal("big")]
└── else_branch
    └── FunctionCall
        ├── function: "print"
        └── arguments: [Literal("small")]
```

The tree clearly shows: "This is an if statement, with a condition, a then branch, and an else branch."

---

## 4c. What is `syn`?

### The Short Answer

**`syn`** is a Rust library (crate) that **parses Rust code into an AST**.

Instead of writing your own parser (which is very hard!), you use `syn` to do the heavy lifting.

### What `syn` Provides

When you give `syn` a string of Rust code, it gives you back Rust data structures representing the AST.

**Example**:
```rust
use syn::{parse_file, Item};

let code = r#"
    fn add(x: i32, y: i32) -> i32 {
        x + y
    }
"#;

// syn parses the code into an AST
let ast = parse_file(code).unwrap();

// Now `ast` is a data structure you can walk through
for item in ast.items {
    match item {
        Item::Fn(func) => {
            println!("Found function: {}", func.sig.ident);
            // You can access func.sig.inputs (parameters)
            // func.sig.output (return type)
            // func.block (function body)
        }
        _ => {}
    }
}

// Output: "Found function: add"
```

### Why Use `syn`?

**Without `syn`**: You'd have to write code to:
- Handle every Rust syntax rule
- Deal with edge cases (generics, lifetimes, macros, etc.)
- Keep up with Rust language changes
- This would take thousands of lines of very complex code!

**With `syn`**: One line does it all:
```rust
let ast = syn::parse_file(code)?;
```

---

## 4d. How Parsing Works: Step by Step

Let's trace how `syn` parses this code:

**Input**:
```rust
fn add(x: i32, y: i32) -> i32 { x + y }
```

### Step 1: Lexing (Tokenization)

First, break the string into **tokens** (meaningful units):

```
Token 1:  Keyword("fn")
Token 2:  Ident("add")
Token 3:  Punct("(")
Token 4:  Ident("x")
Token 5:  Punct(":")
Token 6:  Ident("i32")
Token 7:  Punct(",")
Token 8:  Ident("y")
Token 9:  Punct(":")
Token 10: Ident("i32")
Token 11: Punct(")")
Token 12: Punct("->")
Token 13: Ident("i32")
Token 14: Punct("{")
Token 15: Ident("x")
Token 16: Punct("+")
Token 17: Ident("y")
Token 18: Punct("}")
```

This is like breaking a sentence into words.

### Step 2: Parsing (Building the Tree)

Now, organize these tokens into a tree structure based on Rust's grammar rules:

```
ItemFn {
    sig: Signature {
        ident: "add",
        inputs: [
            FnArg::Typed {
                pat: Ident("x"),
                ty: Type::Path { path: "i32" }
            },
            FnArg::Typed {
                pat: Ident("y"),
                ty: Type::Path { path: "i32" }
            }
        ],
        output: ReturnType::Type(Type::Path { path: "i32" })
    },
    block: Block {
        stmts: [
            Stmt::Expr(
                Expr::Binary {
                    left: Expr::Path { path: "x" },
                    op: BinOp::Add,
                    right: Expr::Path { path: "y" }
                }
            )
        ]
    }
}
```

### Step 3: Working with the AST

Now you can access any part of the code programmatically:

```rust
let ast = syn::parse_file(code)?;

for item in ast.items {
    if let Item::Fn(func) = item {
        // Access function name
        let name = func.sig.ident.to_string();
        println!("Function name: {}", name); // "add"

        // Access parameters
        for param in func.sig.inputs {
            if let FnArg::Typed(pat_type) = param {
                // Get parameter name
                // Get parameter type
            }
        }

        // Access return type
        if let ReturnType::Type(_, ty) = func.sig.output {
            // Work with return type
        }

        // Access function body
        for stmt in func.block.stmts {
            // Work with each statement
        }
    }
}
```

---

## 4e. Parsers in Our Transpiler

### Two Approaches

#### Approach 1: Use `syn` (Production-Quality)

```rust
use syn::parse_file;

pub fn parse_rust_to_ir(rust_code: &str) -> Result<Program, String> {
    // Let syn do the hard work
    let ast = parse_file(rust_code)?;

    // Walk the AST and build our IR
    let mut ir_functions = Vec::new();

    for item in ast.items {
        if let Item::Fn(func) = item {
            // Extract function info from AST
            let ir_function = convert_ast_function_to_ir(func);
            ir_functions.push(ir_function);
        }
    }

    Ok(Program { functions: ir_functions })
}
```

**Pros**: Handles all Rust syntax correctly
**Cons**: More complex code (but we show full example in Appendix)

#### Approach 2: Hardcode (Our Toy Example)

```rust
pub fn parse_rust_simple(rust_code: &str) -> Result<Program, String> {
    // Skip parsing, just create the IR directly
    // We know exactly what the input looks like
    Ok(Program {
        functions: vec![
            Function {
                name: "compute".to_string(),
                // ... hardcoded IR structure
            }
        ]
    })
}
```

**Pros**: Simple, easy to understand
**Cons**: Only works for our specific example

For our **toy example**, we'll use Approach 2 to keep things simple.

For a **real Jolt transpiler**, you'd use something like Approach 1 (or better yet, the "execute and record" approach from zkLean).

---

## 4f. Key Takeaways

Before moving on, make sure you understand:

1. **Parser**: Converts text (string) into structured data (AST)
2. **AST**: Tree-shaped data structure representing code structure
3. **`syn`**: Rust library that parses Rust code into ASTs
4. **Why we need them**: Can't work with code as strings - need structured representation

**Analogy recap**:
- **Code as text** = Recipe written in English
- **Parser** = Someone who reads English and understands the recipe structure
- **AST** = A structured list: "Ingredients: ..., Steps: ..., Time: ..."
- **`syn`** = A professional translator who does this for you

**What's next**: Now that you understand parsers and ASTs, we'll build our toy transpiler!

---

# Part II: Toy Example - Building a Transpiler from Scratch

**Goal**: Show EVERY artifact in the transpilation pipeline with a simple example.

**What we'll build**: A mini-transpiler that converts simple Rust math functions to Go.

## 5. The Complete Pipeline Overview

Here's every file we'll create:

```
toy-transpiler/
├── source.rs           (1) Input: Rust code to transpile
├── ir.rs              (2) Intermediate Representation definitions
├── parser.rs          (3) Parses Rust → IR
├── generator.rs       (4) Generates Go from IR
├── transpiler.rs      (5) Main program (orchestrator)
├── test.rs            (6) Tests that verify equivalence
└── output.go          (7) Generated Go code
```

Let's build each piece step by step.

---

## 6. Step 1: Define the IR

**File: `ir.rs`**

*Recall from Section 3*: The IR is our language-neutral representation. It defines what operations we support (Add, Multiply), not specific programs.

Here's the complete code:

```rust
// ir.rs - Language-neutral representation of our programs

/// A variable (like `x`, `a`, `b`)
#[derive(Debug, Clone)]
pub struct Variable {
    pub name: String,
    pub var_type: Type,
}

/// Types in our mini language
#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Int32,
    // We could add more: Int64, Bool, etc.
}

/// Operations we support
#[derive(Debug, Clone)]
pub enum Operation {
    Add(Variable, Variable),      // a + b
    Multiply(Variable, Variable),  // a * b
    // Could add: Sub, Div, etc.
}

/// A statement in the program
#[derive(Debug, Clone)]
pub enum Statement {
    // let name = operation
    Assignment {
        target: Variable,
        operation: Operation,
    },
    // return variable
    Return(Variable),
}

/// A complete function
#[derive(Debug, Clone)]
pub struct Function {
    pub name: String,
    pub parameters: Vec<Variable>,
    pub return_type: Type,
    pub body: Vec<Statement>,
}

/// A complete program (collection of functions)
#[derive(Debug, Clone)]
pub struct Program {
    pub functions: Vec<Function>,
}
```

**Key points**:
- This defines our **mini language** (only Add and Multiply)
- This file **never changes** when transpiling different programs
- It only changes when we want to support new operation types

---

## 7. Step 2: The Source and Target

**File: `source.rs`** (our input):

```rust
fn compute(x: i32, y: i32) -> i32 {
    let a = x + y;
    let b = a * 2;
    return b;
}
```

**File: `output.go`** (what we want to generate):

```go
func compute(x int32, y int32) int32 {
    a := x + y
    b := a * 2
    return b
}
```

---

## 8. Step 3: Parse Rust → IR

**File: `parser.rs`**

*Recall from Section 4e*: We're using the simplified manual approach (Approach 2) for our toy example.

```rust
// parser.rs - Simplified manual parsing
use crate::ir::*;

pub fn parse_rust_simple(rust_code: &str) -> Result<Program, String> {
    // Hardcoded: we know our input is the compute() function
    // Production would use syn::parse_file() here (see Appendix A)

    Ok(Program {
        functions: vec![
            Function {
                name: "compute".to_string(),
                parameters: vec![
                    Variable { name: "x".to_string(), var_type: Type::Int32 },
                    Variable { name: "y".to_string(), var_type: Type::Int32 },
                ],
                return_type: Type::Int32,
                body: vec![
                    Statement::Assignment {
                        target: Variable { name: "a".to_string(), var_type: Type::Int32 },
                        operation: Operation::Add(
                            Variable { name: "x".to_string(), var_type: Type::Int32 },
                            Variable { name: "y".to_string(), var_type: Type::Int32 },
                        ),
                    },
                    Statement::Assignment {
                        target: Variable { name: "b".to_string(), var_type: Type::Int32 },
                        operation: Operation::Multiply(
                            Variable { name: "a".to_string(), var_type: Type::Int32 },
                            Variable { name: "2".to_string(), var_type: Type::Int32 },
                        ),
                    },
                    Statement::Return(Variable { name: "b".to_string(), var_type: Type::Int32 }),
                ],
            }
        ],
    })
}
```

This creates an IR instance that matches our `compute()` function.

---

## 9. Step 4: Generate Go from IR

**File: `generator.rs`**

The generator is a **template engine** - it converts IR data structures into Go text strings.

Think of it like `println!` for code generation.

### How It Works: Concrete Example

**Input IR** (data structure in memory):
```rust
Statement::Assignment {
    target: Variable { name: "a", var_type: Int32 },
    operation: Operation::Add(
        Variable { name: "x", var_type: Int32 },
        Variable { name: "y", var_type: Int32 }
    )
}
```

**Output Go** (text string):
```go
"a := x + y"
```

### The Complete Generator Code

```rust
// generator.rs - Convert IR to Go code
use crate::ir::*;

pub fn generate_go(program: &Program) -> String {
    let mut output = String::new();

    // Start with package declaration
    output.push_str("package main\n\n");

    // Generate each function
    for function in &program.functions {
        output.push_str(&generate_function(function));
        output.push_str("\n\n");
    }

    output  // Return the complete Go code as a String
}

fn generate_function(func: &Function) -> String {
    let mut output = String::new();

    // Generate: "func add(x int32, y int32) int32 {"
    output.push_str("func ");
    output.push_str(&func.name);
    output.push_str("(");

    // Generate parameters: "x int32, y int32"
    let params: Vec<String> = func.parameters.iter()
        .map(|p| format!("{} {}", p.name, type_to_go(&p.var_type)))
        .collect();
    output.push_str(&params.join(", "));

    output.push_str(") ");
    output.push_str(&type_to_go(&func.return_type));
    output.push_str(" {\n");

    // Generate function body statements
    for stmt in &func.body {
        output.push_str("    ");  // Indentation
        output.push_str(&generate_statement(stmt));
        output.push_str("\n");
    }

    output.push_str("}");

    output
}

fn generate_statement(stmt: &Statement) -> String {
    match stmt {
        Statement::Assignment { target, operation } => {
            // IR: Assignment { target: "a", operation: Add(x, y) }
            // Go: "a := x + y"
            format!("{} := {}", target.name, generate_operation(operation))
        }
        Statement::Return(var) => {
            // IR: Return(Variable { name: "result" })
            // Go: "return result"
            format!("return {}", var.name)
        }
    }
}

fn generate_operation(op: &Operation) -> String {
    match op {
        Operation::Add(left, right) => {
            // IR: Add(Variable("x"), Variable("y"))
            // Go: "x + y"
            format!("{} + {}", left.name, right.name)
        }
        Operation::Multiply(left, right) => {
            // IR: Multiply(Variable("a"), Variable("b"))
            // Go: "a * b"
            format!("{} * {}", left.name, right.name)
        }
    }
}

fn type_to_go(ty: &Type) -> &str {
    match ty {
        Type::Int32 => "int32",
        // If we add Type::Int64, we'd add:
        // Type::Int64 => "int64",
    }
}
```

### Why No Blueprint Exists

**Question**: "Can't we use a library for IR → Go conversion?"

**Answer**: No, because:

1. **Your IR is custom**: Only YOU defined `Operation::Add` and `Operation::Multiply`
2. **Mapping is custom**: Only YOU decide that `Add(x, y)` becomes `"x + y"` in Go
3. **Target syntax is custom**: You could target Go, C++, JavaScript, etc.

**It's just string building** - no magic library can do it for you.

### Analogy: Mad Libs

The generator is like Mad Libs:

**Template** (what you want to generate):
```
"func [NAME]([PARAMS]) [RETURN_TYPE] {
    [BODY]
}"
```

**Fill in the blanks** (from your IR):
- `[NAME]` = `func.name` → `"compute"`
- `[PARAMS]` = `func.parameters` → `"x int32, y int32"`
- `[RETURN_TYPE]` = `func.return_type` → `"int32"`
- `[BODY]` = `func.body` → `"a := x + y\n    b := a * 2\n    return b"`

**Result**:
```go
func compute(x int32, y int32) int32 {
    a := x + y
    b := a * 2
    return b
}
```

### Trace Through an Example

Let's trace how this IR becomes Go code:

**IR**:
```rust
Function {
    name: "compute",
    parameters: [
        Variable { name: "x", var_type: Int32 },
        Variable { name: "y", var_type: Int32 }
    ],
    return_type: Int32,
    body: [
        Assignment {
            target: Variable { name: "a", var_type: Int32 },
            operation: Add(
                Variable { name: "x", var_type: Int32 },
                Variable { name: "y", var_type: Int32 }
            )
        }
    ]
}
```

**Step 1**: `generate_function()` starts
```rust
output = "func "
output += "compute"      // → "func compute"
output += "("            // → "func compute("
```

**Step 2**: Generate parameters
```rust
// For each parameter, create "name type"
params = ["x int32", "y int32"]
output += "x int32, y int32"  // → "func compute(x int32, y int32"
output += ") "                 // → "func compute(x int32, y int32) "
output += "int32"              // → "func compute(x int32, y int32) int32"
output += " {\n"               // → "func compute(x int32, y int32) int32 {\n"
```

**Step 3**: Generate body
```rust
// Process the Assignment statement
stmt_str = generate_statement(body[0])
// Returns: "a := x + y"
output += "    "       // Indentation
output += "a := x + y"
output += "\n"
// Result: "func compute(...) int32 {\n    a := x + y\n"
```

**Step 4**: Close function
```rust
output += "}"
// Final: "func compute(x int32, y int32) int32 {\n    a := x + y\n}"
```

### Key Insight

**The generator is just string concatenation with pattern matching**:
- Match on your IR enums
- Build strings for each case
- Concatenate them together

No magic, no library - just building text!

---

## 10. Step 5: Main Transpiler Program

**File: `transpiler.rs`**

```rust
// transpiler.rs - Main program that ties everything together
mod ir;
mod parser;
mod generator;

use std::fs;

fn main() -> Result<(), String> {
    // Step 1: Read source Rust file
    let rust_code = fs::read_to_string("source.rs")
        .map_err(|e| format!("Cannot read source.rs: {}", e))?;

    println!("📖 Reading source.rs...");
    println!("{}\n", rust_code);

    // Step 2: Parse Rust → IR
    println!("🔍 Parsing Rust to IR...");
    let ir = parser::parse_rust_simple(&rust_code)?;

    println!("IR created:");
    println!("{:#?}\n", ir);

    // Step 3: Generate Go from IR
    println!("⚙️  Generating Go code...");
    let go_code = generator::generate_go(&ir);

    println!("Generated Go:");
    println!("{}\n", go_code);

    // Step 4: Write output
    fs::write("output.go", &go_code)
        .map_err(|e| format!("Cannot write output.go: {}", e))?;

    println!("✅ Transpilation complete! Check output.go");

    Ok(())
}
```

**What it does**:
```
Read source.rs → Parse to IR → Generate Go → Write output.go
```

---

## 11. Step 6: Testing Equivalence

**File: `test.rs`**

```rust
// test.rs - Verify Rust and Go produce same results

#[test]
fn test_compute_equivalence() {
    // Test case 1
    let x1 = 5;
    let y1 = 3;
    let rust_result1 = compute_rust(x1, y1);
    let go_result1 = run_go_program("output.go", x1, y1);
    assert_eq!(rust_result1, go_result1);

    // Test case 2
    let x2 = 10;
    let y2 = 20;
    let rust_result2 = compute_rust(x2, y2);
    let go_result2 = run_go_program("output.go", x2, y2);
    assert_eq!(rust_result2, go_result2);
}

// Original Rust implementation
fn compute_rust(x: i32, y: i32) -> i32 {
    let a = x + y;
    let b = a * 2;
    return b;
}

// Helper: Run Go program and capture output
fn run_go_program(path: &str, x: i32, y: i32) -> i32 {
    use std::process::Command;

    // Create a test main.go that calls our function
    let test_go = format!(r#"
package main

import "fmt"

func main() {{
    result := compute({}, {})
    fmt.Print(result)
}}
"#, x, y);

    std::fs::write("test_main.go", test_go).unwrap();

    // Run: go run test_main.go output.go
    let output = Command::new("go")
        .args(&["run", "test_main.go", "output.go"])
        .output()
        .expect("Failed to run Go");

    let result_str = String::from_utf8(output.stdout).unwrap();
    result_str.trim().parse::<i32>().unwrap()
}
```

**What it does**: Runs both Rust and Go versions with same inputs, verifies same outputs.

This is called **differential testing** - critical for correctness!

---

## 12. Running the Example

```bash
# 1. Create source file
cat > source.rs << 'EOF'
fn compute(x: i32, y: i32) -> i32 {
    let a = x + y;
    let b = a * 2;
    return b;
}
EOF

# 2. Run transpiler
cargo run --bin transpiler

# Output:
# 📖 Reading source.rs...
# 🔍 Parsing Rust to IR...
# IR created: Program { ... }
# ⚙️  Generating Go code...
# Generated Go: func compute(x int32, y int32) int32 { ... }
# ✅ Transpilation complete!

# 3. Test the generated code
cargo test

# Output:
# test test_compute_equivalence ... ok
```

---

## 13. All Artifacts Summary

| Artifact | Type | Purpose |
|----------|------|---------|
| **source.rs** | Input | Original Rust code |
| **ir.rs** | Code | IR type definitions (NEVER changes per program) |
| **parser.rs** | Code | Rust → IR converter |
| **generator.rs** | Code | IR → Go converter |
| **transpiler.rs** | Code | Main orchestrator |
| **test.rs** | Testing | Equivalence verification |
| **output.go** | Output | Generated Go code |
| **IR instance** | Runtime data | Parsed program (changes per program) |

**The key insight**: The IR definition (`ir.rs`) is the bridge - it stays stable while we transpile many different programs.

---

## 14. Key Takeaways from Toy Example

1. **Transpilation = Parse → IR → Generate**
2. **IR is language-neutral**: Not Rust, not Go, just data structures
3. **Parser extracts structure**: Reads source, builds IR instances
4. **Generator builds target**: Reads IR, emits code
5. **Testing verifies correctness**: Differential testing is critical
6. **IR definition is stable**: Only changes when adding new operation types

---

# Part III: Applying Transpilation to Jolt → Gnark

Now that you understand transpilation, let's apply it to the real problem: converting Jolt's Rust verifier to Gnark circuits in Go.

## 15. The Jolt-Gnark Challenge

### 15.1 The Goal

Instead of:
```
Manually rewrite Jolt verifier in Go (Gnark DSL)
    ↓
Maintain two codebases (Rust + Go)
    ↓
Manually sync when Jolt protocol changes
```

We want:
```
Jolt Rust Verifier (single source of truth)
    ↓
Extraction Tool (programmatic)
    ↓
Gnark Circuit Generation (automatic)
    ↓
Update when Jolt changes by re-running extraction
```

### 15.2 Key Differences from Toy Example

| Aspect | Toy Example | Jolt → Gnark |
|--------|-------------|--------------|
| **Source language** | Rust (general code) | Rust (cryptographic operations) |
| **Target language** | Go (general code) | Gnark circuit DSL (constraints) |
| **Operations** | Add, multiply integers | Field arithmetic, polynomial evaluation, sumcheck |
| **Extraction method** | Parse text | Execute and record operations |
| **Complexity** | 2 operations, 1 type | ~10 operation types, complex polynomials |

### 15.3 Why Not Just Use Our Toy Transpiler?

**Target is different**: Gnark is not "normal Go" - it's a Domain-Specific Language (DSL) for circuits.

```go
// This isn't normal Go - it's Gnark's circuit DSL
func (circuit *MyCircuit) Define(api frontend.API) error {
    // These aren't regular Go operations - they create R1CS constraints
    x := api.Add(circuit.A, circuit.B)      // Creates constraint
    y := api.Mul(x, circuit.C)              // Creates another constraint
    api.AssertIsEqual(y, circuit.Output)    // Adds assertion constraint
    return nil
}
```

Each `api.Add()`, `api.Mul()` call **generates R1CS constraints** - it's not computing values, it's building a constraint system.

We're targeting **Gnark's constraint-building API**, not arbitrary Go code.

---

## 16. The zkLean Precedent

Before designing our own system, let's learn from PR #1060 which does exactly this pattern!

### 16.1 What is zkLean?

**Goal**: Formally verify that Jolt's cryptographic protocol is correct (no bugs, no security holes)

**Challenge**: Formal verification requires writing specifications in Lean4 (a proof language)
- Writing these by hand is tedious and error-prone
- Keeping them in sync with Rust code is a nightmare

**Solution**: Extract the specification directly from Rust code, generate Lean4 automatically

### 16.2 zkLean's Extraction Pattern

```
Jolt Rust Implementation
    ↓
Trait/Builder Instantiation (e.g., R1CSBuilder, JoltLookupTable)
    ↓
Algebraic Data Structures (Constraint, MleAst, etc.)
    ↓
Target Language Generation (Lean4 syntax)
```

**Critical insight**: The Rust code is the single source of truth. The extractor doesn't manually "translate"—it executes Jolt's logic in a constrained context that captures operations as data.

### 16.3 What zkLean Extracts

Looking at `zklean-extractor/src/main.rs`, four modules are extracted:

1. **R1CS Constraints** (`src/r1cs.rs`): The ~30 constraints applied to every cycle
2. **Subtables** (`src/subtable.rs`): MLE evaluations for lookup tables
3. **Instructions** (`src/instruction.rs`): RISC-V instruction specifications
4. **Lookup Cases** (`src/flags.rs`): Circuit flags and decomposition logic

**How it works**:
```rust
pub fn extract() -> Self {
    let inputs = JoltR1CSInputs::flatten::<{ J::C }>();

    let uniform_constraints = {
        let mut r1cs_builder = R1CSBuilder::<{ J::C }, F, JoltR1CSInputs>::new();
        CS::uniform_constraints(&mut r1cs_builder, memory_layout.input_start);
        r1cs_builder.get_constraints()  // ← Extract the recorded operations
    };

    Self { inputs, uniform_constraints, ... }
}
```

**Key technique**: Instantiate Jolt's R1CS builder, call the constraint generation functions, extract the resulting constraints as data structures.

### 16.4 Key Lessons from zkLean

**What it proves**:
- ✅ Extraction from Jolt's Rust code works
- ✅ Automatic generation of target language works
- ✅ The pattern is maintainable (updates are automatic)
- ✅ Differential testing catches bugs

**What it warns about**:
- ⚠️ Scaling requires optimization (arena allocation, common subexpression elimination)
- ⚠️ Target language matters (Lean had stack overflow issues on large circuits)
- ⚠️ Engineering investment needed

---

## 17. Jolt IR Design

Based on zkLean's pattern, here's what a Jolt IR might look like:

```rust
// jolt_ir.rs - Hypothetical IR for Jolt verification
// This SINGLE FILE would support transpiling ALL Jolt verification stages

/// Field elements (arithmetic modulo prime)
#[derive(Debug, Clone)]
pub struct FieldElement {
    pub name: String,
}

/// Polynomials (multilinear extensions)
#[derive(Debug, Clone)]
pub struct Polynomial {
    pub name: String,
    pub coefficients: Vec<FieldElement>,
}

/// Field operations (basic arithmetic in the field)
#[derive(Debug, Clone)]
pub enum FieldOperation {
    Add(FieldElement, FieldElement),
    Multiply(FieldElement, FieldElement),
    Exponentiation(FieldElement, u64),
    Inverse(FieldElement),
}

/// Polynomial operations
#[derive(Debug, Clone)]
pub enum PolynomialOperation {
    Evaluate {
        poly: Polynomial,
        point: FieldElement,
    },
    Commitment {
        poly: Polynomial,
    },
}

/// Sumcheck round (from the sumcheck protocol)
#[derive(Debug, Clone)]
pub struct SumcheckRound {
    pub round_number: usize,
    pub prover_message: Vec<FieldElement>,  // Univariate polynomial coefficients
    pub verifier_challenge: FieldElement,
}

/// Circuit constraints (what must be true)
#[derive(Debug, Clone)]
pub enum Constraint {
    Equality {
        left: FieldElement,
        right: FieldElement,
    },
    SumEquals {
        terms: Vec<FieldElement>,
        sum: FieldElement,
    },
}

/// Statements in the verification circuit
#[derive(Debug, Clone)]
pub enum Statement {
    FieldOp {
        target: FieldElement,
        operation: FieldOperation,
    },
    PolyOp {
        target: FieldElement,
        operation: PolynomialOperation,
    },
    SumcheckStep {
        round: SumcheckRound,
    },
    Assert {
        constraint: Constraint,
    },
}

/// A complete verification circuit
#[derive(Debug, Clone)]
pub struct VerificationCircuit {
    pub name: String,
    pub public_inputs: Vec<FieldElement>,
    pub statements: Vec<Statement>,
}

/// The complete program (all verification stages)
#[derive(Debug, Clone)]
pub struct JoltVerifierProgram {
    pub circuits: Vec<VerificationCircuit>,
}
```

**This single IR could represent all Jolt verification stages**:
- Stage 1: R1CS verification
- Stage 2-4: Sumcheck verification
- Stage 5: Dory opening proof
- Stage 6: Hyrax verification (with manual elliptic curve gadgets)

---

## 18. Extraction Strategy

### 18.1 How Extraction Differs from Parsing

For our toy example, we used **parsing** (syn reads Rust text → AST → IR).

For Jolt, we use **execute and record** (run Jolt code with special types that log operations).

**Why the difference?** Because Jolt's verifier code is too complex to parse!

---

#### Approach A: Parsing (What We Did in Toy Example)

**How it works**:
```rust
// Read Rust source code as text
let rust_code = read_file("jolt-core/src/subprotocols/sumcheck.rs")?;

// Use syn to parse text → AST
let ast = syn::parse_file(rust_code)?;

// Walk AST, extract operations → IR
for item in ast.items {
    // Figure out what operations the code does
}
```

**Why this is HARD for Jolt**:
1. **Complex Rust**: Jolt uses traits, generics, lifetimes, macros
2. **Indirect operations**: Operations hidden behind trait implementations
3. **Dynamic behavior**: What happens depends on runtime values
4. **Maintenance nightmare**: If Jolt refactors code, parser breaks

**Example of complexity**:
```rust
// Jolt's actual code (simplified)
pub fn verify<F, CS>(builder: &mut CS, proof: &Proof<F>)
where
    F: JoltField,
    CS: ConstraintSystem<F>
{
    let claim = builder.mul(proof.a, proof.b);
    let eval = self.evaluate(&proof.challenge);
    builder.assert_equal(claim, eval);
}
```

Parsing this requires understanding:
- Generic type parameters (`F`, `CS`)
- Trait bounds (`JoltField`, `ConstraintSystem`)
- What `builder.mul()` does (depends on trait implementation)
- What `self.evaluate()` does (complex polynomial logic)

**This is too hard!** We'd need to re-implement Rust's type system.

---

#### Approach B: Execute and Record (What zkLean/Jolt Uses)

**How it works**: Instead of parsing text, **run the actual Jolt code** with special "recording" types.

```rust
// Step 1: Create a builder that RECORDS operations instead of computing
let mut builder = RecordingR1CSBuilder::new();

// Step 2: Call Jolt's ACTUAL function (not parsing, actually running it!)
JoltConstraints::uniform_constraints(&mut builder);
// This runs Jolt's real code, which calls builder.add(), builder.mul(), etc.

// Step 3: Extract what operations were recorded
let operations = builder.get_recorded_constraints();
// Now we have a list of operations that Jolt performed!
```

**The magic**: `RecordingR1CSBuilder` implements the same trait as Jolt's normal builder, but instead of computing, it **logs** what happened.

**Concrete example**:

```rust
// Normal R1CS builder (used in actual proving)
impl R1CSBuilder {
    fn mul(&mut self, a: Variable, b: Variable) -> Variable {
        // Actually compute a * b in the field
        let result = a.value * b.value;
        // Add constraint to R1CS
        self.constraints.push(Constraint::Mul(a, b, result));
        result
    }
}

// Recording R1CS builder (used for extraction)
impl RecordingR1CSBuilder {
    fn mul(&mut self, a: Variable, b: Variable) -> Variable {
        // DON'T compute anything!
        // Just RECORD that a multiplication happened
        self.recorded_ops.push(Operation::Multiply(a, b));

        // Return a dummy variable (value doesn't matter)
        Variable::new_dummy()
    }
}
```

Now when Jolt's code runs:
```rust
// Jolt's actual code
let result = builder.mul(x, y);
```

With `RecordingR1CSBuilder`:
- **Doesn't actually multiply** x and y
- **Records**: "Operation: Multiply(x, y)"
- **Returns** dummy variable (so code keeps running)

---

#### Why "Execute and Record" is Better for Jolt

| Aspect | Parsing (syn) | Execute and Record |
|--------|---------------|-------------------|
| **Complexity** | Must understand all Rust syntax | Just provide recording types |
| **Generics/Traits** | Must resolve type parameters | Rust compiler handles it |
| **Maintenance** | Breaks when Jolt code changes | Automatically adapts |
| **Correctness** | Might misinterpret code | Uses Jolt's actual logic |
| **Implementation** | ~5000 lines of parsing logic | ~500 lines of recording types |

**Example**: If Jolt refactors from:
```rust
let x = a.mul(b);
```
to:
```rust
let x = a * b;  // Using operator overloading
```

- **Parsing approach**: Parser breaks! Need to rewrite AST walking logic
- **Recording approach**: Still works! `*` operator calls `mul()` trait method, which we intercept

---

#### Concrete Example: Recording a Sumcheck Round

**Jolt's actual verifier code** (simplified):
```rust
pub fn verify_round(
    transcript: &mut Transcript,
    proof: &RoundProof,
) -> Result<FieldElement> {
    // Read challenge from transcript
    let challenge = transcript.read_challenge();

    // Verify g(0) + g(1) = claimed_sum
    let g_0 = proof.poly.evaluate(FieldElement::zero());
    let g_1 = proof.poly.evaluate(FieldElement::one());
    let sum = g_0 + g_1;

    assert_eq!(sum, proof.claimed_sum);

    // Compute g(challenge)
    proof.poly.evaluate(challenge)
}
```

**Extraction with recording types**:
```rust
// Step 1: Create recording transcript
let mut recording_transcript = RecordingTranscript::new();

// Step 2: Create fake proof (structure matters, values don't)
let fake_proof = RoundProof {
    poly: DensePolynomial::new_dummy(),
    claimed_sum: FieldElement::dummy(),
};

// Step 3: Run Jolt's ACTUAL verifier code
let result = verify_round(&mut recording_transcript, &fake_proof);

// Step 4: Extract what operations were recorded
let ir = recording_transcript.get_operations();
// ir now contains:
// [
//   Operation::ReadChallenge,
//   Operation::PolyEval(poly, 0),
//   Operation::PolyEval(poly, 1),
//   Operation::Add(g_0, g_1),
//   Operation::AssertEqual(sum, claimed_sum),
//   Operation::PolyEval(poly, challenge),
// ]
```

**What we recorded**:
1. Reading a challenge from transcript
2. Two polynomial evaluations (at 0 and 1)
3. Addition
4. Assertion
5. Another polynomial evaluation (at random challenge)

**Now generate Gnark**:
```go
func VerifySumcheckRound(api frontend.API, proof RoundProof) frontend.Variable {
    challenge := ReadChallenge(api)
    g_0 := EvaluatePolynomial(api, proof.Poly, 0)
    g_1 := EvaluatePolynomial(api, proof.Poly, 1)
    sum := api.Add(g_0, g_1)
    api.AssertIsEqual(sum, proof.ClaimedSum)
    return EvaluatePolynomial(api, proof.Poly, challenge)
}
```

---

#### Why You CAN'T Use syn for Jolt

**You could technically use syn**, but you'd have to:

1. **Parse Rust code** (syn can do this)
2. **Resolve all types** (syn doesn't do this - you'd need to reimplement rustc's type checker!)
3. **Understand trait implementations** (which `builder.mul()` is being called?)
4. **Handle macros** (expand them manually)
5. **Interpret control flow** (what does this loop do?)

This is essentially **writing a Rust compiler**. Way too hard!

**Execute and record** lets Rust's compiler do all the hard work, and you just log what happens.

---

#### Summary

**Question**: "Why is execute-and-record better? Can't we use syn?"

**Answer**:

**Technically yes**, but practically no:
- **syn can parse** Jolt's Rust code into an AST
- **But you still need** to interpret that AST (resolve types, traits, generics, macros)
- **That's incredibly hard** - you'd be reimplementing much of rustc

**Execute-and-record is better because**:
- Rust compiler handles all the complexity
- You just provide recording types
- Jolt's actual code runs (guaranteed correct)
- Automatically adapts when Jolt changes
- Much less code (~500 lines vs ~5000+ lines)

### 18.2 Example: Extracting Sumcheck

```rust
// In jolt-core-extractor crate
pub struct SumcheckExtractor;

impl SumcheckExtractor {
    pub fn extract_round_verification() -> VerifierCircuit {
        // Create recording transcript
        let mut recording_transcript = RecordingTranscript::new();

        // Create fake proof data (structure matters, values don't)
        let fake_proof = generate_template_proof();

        // Run verifier in "trace mode"
        let result = jolt_core::sumcheck::verify_round(
            &fake_proof,
            &mut recording_transcript
        );

        // Extract recorded operations
        VerifierCircuit {
            inputs: recording_transcript.get_inputs(),
            operations: recording_transcript.get_operations(),
            outputs: recording_transcript.get_outputs(),
        }
    }
}
```

---

## 19. Code Generation for Gnark

Once we have the IR, generate Gnark circuits:

```rust
pub struct GnarkGenerator;

impl GnarkGenerator {
    pub fn generate(&self, circuit: &VerifierCircuit) -> Result<String> {
        let mut output = String::new();

        // Package declaration
        output.push_str("package jolt_verifier\n\n");
        output.push_str("import \"github.com/consensys/gnark/frontend\"\n\n");

        // Generate circuit struct
        output.push_str(&self.generate_circuit_struct(&circuit.inputs));

        // Generate Define() method
        output.push_str(&self.generate_define_method(&circuit.operations));

        Ok(output)
    }

    fn generate_define_method(&self, operations: &[Statement]) -> String {
        let mut output = String::new();

        output.push_str("func (circuit *JoltVerifierCircuit) Define(api frontend.API) error {\n");

        for op in operations {
            output.push_str(&self.generate_statement(op));
        }

        output.push_str("    return nil\n");
        output.push_str("}\n");

        output
    }

    fn generate_statement(&self, stmt: &Statement) -> String {
        match stmt {
            Statement::FieldOp { target, operation } => {
                match operation {
                    FieldOperation::Add(left, right) => {
                        format!("    {} := api.Add({}, {})\n", target.name, left.name, right.name)
                    }
                    FieldOperation::Multiply(left, right) => {
                        format!("    {} := api.Mul({}, {})\n", target.name, left.name, right.name)
                    }
                    // ... other operations
                }
            }
            Statement::Assert { constraint } => {
                match constraint {
                    Constraint::Equality { left, right } => {
                        format!("    api.AssertIsEqual({}, {})\n", left.name, right.name)
                    }
                    // ... other constraints
                }
            }
            // ... other statement types
        }
    }
}
```

**Output example**:
```go
// Auto-generated by jolt-to-gnark extractor
package jolt_verifier

import "github.com/consensys/gnark/frontend"

type JoltVerifierCircuit struct {
    // Public inputs
    ProofCommitments [29]frontend.Variable `gnark:",public"`

    // Witness (private)
    UnivariateCoeffs [][]frontend.Variable
    // ... other proof elements
}

func (circuit *JoltVerifierCircuit) Define(api frontend.API) error {
    // Stage 1: Sumcheck verification
    claim_0 := circuit.verifySumcheck_0(api)
    claim_1 := circuit.verifySumcheck_1(api)
    // ...

    return nil
}
```

---

## 20. Comparison: Full Rewrite vs Partial Extraction

### 20.0 Industry Research Findings (November 2025)

Before comparing approaches, let's examine what SP1 and RISC Zero actually do:

#### **RISC Zero: Fully Manual Circom**

| Aspect | Details |
|--------|---------|
| **Approach** | Fully manual, hand-written Circom |
| **Circuit File** | `stark_verify.circom` - **55.7 MB** (stored in Git LFS) |
| **Pipeline** | STARK proof → Circom circuit → snarkjs → Groth16 proof |
| **Trusted Setup** | Hermez rollup with 2^23 powers of tau |
| **Platform Limitation** | x86-only (Circom witness generator uses x86 assembly) |

**Why manual works for RISC Zero**: Their STARK verification protocol is mature and stable. The enormous 55MB Circom file is a one-time investment that rarely needs updates.

#### **SP1: Semi-Automatic via Recursion DSL**

| Aspect | Details |
|--------|---------|
| **Approach** | Semi-automatic via **Recursion DSL Compiler** |
| **Architecture** | STARK proof → Recursion DSL → Gnark circuit → Groth16 proof |
| **Circuit Generation** | **Precompiled circuits** ("hot start") - ~18s pure prove time |
| **FFI** | `sp1-recursion-gnark-ffi` crate for Rust→Go interop |
| **Maintenance** | DSL program changes → circuit auto-regenerates |

**Key insight**: SP1 built a **custom DSL compiler** that translates their recursion program to Gnark circuits. This validates the "compile from higher representation" approach.

#### **Industry Comparison Matrix**

| Aspect | RISC Zero | SP1 | Jolt (proposed) |
|--------|-----------|-----|-----------------|
| **Automation** | ❌ Manual | ✅ Semi-automatic (DSL) | ✅ Automatic (zkLean) |
| **Circuit Language** | Circom | Custom DSL → Gnark | Rust → AST → Gnark |
| **Maintenance** | Manual updates | DSL program updates | Re-run extractor |
| **Protocol Stability** | Stable | Evolving | Evolving |
| **Existing Infra** | N/A | Built custom compiler | Reuse zkLean |

**Validation**: SP1's approach validates our strategy - they faced the same problem (evolving protocol) and solved it with automatic generation from a higher-level representation. We can do the same by reusing zkLean's existing extraction infrastructure.

### 20.1 Full Rewrite (SP1/Risc0 Approach)

**How it works**:
- Manually implement verifier in Gnark
- Circuit engineers write Go code that mirrors verification
- Optimize circuit-specific details
- Maintain separately from main proving system

**Timeline**: 6-12 months

**Pros**:
- Full control over circuit optimization
- Can leverage Gnark best practices
- Proven pattern (production use)

**Cons**:
- High initial engineering cost
- Divergence risk (Rust verifier ≠ Go circuit)
- Every protocol change requires manual circuit update
- Testing burden (ensure equivalence)
- **RISC Zero's 55MB Circom file demonstrates the maintenance burden**

### 20.2 Partial Extraction (Proposed)

**How it works**:
- Extract core verification logic automatically
- Generate Gnark circuits from IR
- Manually implement complex components (elliptic curves)
- Rerun extraction when Jolt changes

**Timeline**: 6-9 months (including tooling)

**Pros**:
- Single source of truth (Rust implementation)
- Automatic sync with protocol changes
- Reduced error rate (differential testing)
- Flexible (can target multiple backends)

**Cons**:
- Upfront tooling investment
- IR design complexity
- May miss optimization opportunities
- Complex components still manual

### 20.3 Side-by-Side Comparison

| Aspect | Full Rewrite | Partial Extraction |
|--------|-------------|-------------------|
| **Initial effort** | 6-12 months | 6-9 months (including tooling) |
| **Maintenance** | Manual sync (weeks per update) | Automatic (hours to rerun) |
| **Error rate** | Higher (manual translation) | Lower (differential testing) |
| **Optimization** | Full control | Mixed (auto + manual) |
| **Flexibility** | Single target (Gnark) | Multiple targets possible |
| **Learning curve** | Gnark expertise | Extractor framework |

---

## 21. Proof of Concept Scope

To validate this approach, we propose a minimal PoC:

### 21.1 PoC Goal

Extract and generate circuit for **single sumcheck round verification**

**Scope**:
- Input: Univariate polynomial coefficients, claimed sum, challenge
- Operation: Verify $g(0) + g(1) = \text{claim}$ and compute $g(r)$
- Output: Next round's claim value

### 21.2 PoC Success Criteria

- ✅ Extraction tool successfully captures sumcheck round logic
- ✅ Generated Gnark circuit compiles
- ✅ Differential testing passes (100 random test cases)
- ✅ Constraint count measured (<1000 constraints for single round)

### 21.3 PoC Timeline

- Week 1-2: Build extraction framework
- Week 3: Implement IR and Gnark code generator
- Week 4: Testing and validation
- **Total: 4 weeks for minimal PoC**

---

## 22. Technical Challenges & Solutions

### 22.1 Challenge: Control Flow

**Problem**: Circuits are static; Rust verifier has dynamic control flow (loops)

**Solution**: Unroll loops with known bounds
```rust
// Rust: dynamic loop
for proof_elem in proof.iter() {
    verify_element(proof_elem);
}

// Extracted as: fixed unrolling
verify_element(proof[0]);
verify_element(proof[1]);
// ... up to proof[39]
```

### 22.2 Challenge: Witness vs Public Input

**Problem**: Gnark needs clear separation of public inputs vs private witness

**Solution**:
- Annotate Jolt proof structure with `pub/witness` tags
- Extractor maps to Gnark's `Public`/`Secret` fields
- Public: Commitments, challenges, public outputs
- Witness: Polynomial coefficients, intermediate values

### 22.3 Challenge: Cryptographic Primitives

**Problem**: Stage 6 Hyrax verification involves complex elliptic curve operations

**Solution**: Hybrid approach
- Extract high-level "verify Hyrax opening" operation
- Manually implement efficient Gnark gadgets for Grumpkin operations
- Compose extracted logic with manual gadgets

---

## 23. Conclusion & Recommendation

### 23.1 Why Partial Extraction is Promising

The zkLean precedent demonstrates:
- ✅ **Feasible**: Extracting algebraic logic from Rust works
- ✅ **Maintainable**: Single source of truth, automatic sync
- ✅ **Testable**: Differential testing ensures correctness

Applied to Groth16 conversion:
- Core sumcheck logic is algebraic (good fit for extraction)
- Jolt's verifier is well-structured (clear stages, batching)
- Main challenges (elliptic curves) can be handled with hybrid approach

### 23.2 Recommended Next Steps

1. **Validate PoC** (4 weeks)
   - Build minimal extractor for single sumcheck round
   - Generate Gnark circuit
   - Measure constraints, test equivalence

2. **If PoC succeeds** → Full implementation
   - Extend extractor to all sumcheck types
   - Implement hybrid approach for Stage 6
   - Comprehensive testing and optimization

3. **If PoC fails** → Pivot to full rewrite
   - But: Learn from PoC what aspects are extractable
   - Use partial extraction for testable components

---

## Appendix A: Why Not Automatic Rust-to-Circuit Compilers?

### A.1 The Appeal of Automatic Compilation

The ideal approach would be: write Rust, automatically get a circuit. Several projects attempt this:

| Tool | Output Format | Status |
|------|---------------|--------|
| **zkLLVM** | PLONK (Placeholder) | Rust repo archived Feb 2025 |
| **Lurk** | Nova/SuperNova | Lisp-based, not Rust |
| **Nexus** | Nova/Folding | Own VM, not direct transpilation |
| **Arkworks** | R1CS | Manual constraint writing only |

### A.2 zkLLVM Investigation

We investigated zkLLVM as a potential automatic approach:

**What zkLLVM does**:
- LLVM-based circuit compiler
- Compile C++/Rust → LLVM IR → Circuit
- Used by =nil; Foundation for zkProof generation

**Critical blocker discovered**:
- **zkLLVM outputs PLONK (Placeholder), NOT R1CS**
- Groth16 requires R1CS format
- These are fundamentally different arithmetization approaches
- No straightforward conversion between PLONK and R1CS

**Additional concerns**:
- zkLLVM's Rust repository ([zkllvm-rslang](https://github.com/nickvergessen/zkllvm-rslang)) was **archived in February 2025**
- Active development focused on C++ frontend
- Complex LLVM integration required

### A.3 The Fundamental Trade-off

| Approach | Automation Level | Output Format |
|----------|------------------|---------------|
| **zkLLVM/Noir/Leo** | ✅ Automatic | PLONK (not R1CS) |
| **zkVMs (RISC0, SP1)** | ✅ Automatic | STARK → Groth16 wrapper |
| **Arkworks manual** | ❌ Manual | R1CS |
| **zkLean transpilation** | ✅ Automatic | R1CS (via Gnark) |

**Conclusion**: There is no existing tool for automatic Rust → R1CS conversion. This is why the zkLean-based transpilation approach is the only viable path for:
1. **Automatic** generation (not manual rewrite)
2. **R1CS output** (required for Groth16)
3. **Maintainability** (re-run when Jolt changes)

### A.4 Why zkLean Approach Works

The zkLean approach sidesteps the "automatic compiler" problem by:

1. **Not parsing Rust** - Instead, execute Rust with recording types
2. **Capturing semantics, not syntax** - Records what operations happen, not how they're written
3. **Reusing Jolt's type system** - Rust compiler handles all complexity
4. **Targeting constraint-friendly IR** - MleAst already represents algebraic operations

This is similar to how SP1 built their Recursion DSL Compiler - they didn't try to compile arbitrary Rust, they built a targeted extraction from their specific domain.

---

## Appendix B: Full Parser Example (Using syn)

For reference, here's how you'd parse Rust using the `syn` crate:

```rust
// parser.rs - Using syn to parse Rust
use syn::{parse_file, Item, Stmt, Expr};
use crate::ir::*;

pub fn parse_rust_to_ir(rust_code: &str) -> Result<Program, String> {
    // Parse Rust code into Abstract Syntax Tree (AST)
    let ast = parse_file(rust_code)
        .map_err(|e| format!("Parse error: {}", e))?;

    let mut functions = Vec::new();

    // Walk through the AST
    for item in ast.items {
        match item {
            Item::Fn(func) => {
                let name = func.sig.ident.to_string();

                // Extract parameters
                let mut parameters = Vec::new();
                for param in func.sig.inputs {
                    if let syn::FnArg::Typed(pat_type) = param {
                        let param_name = extract_param_name(&pat_type.pat);
                        let param_type = extract_type(&pat_type.ty);
                        parameters.push(Variable {
                            name: param_name,
                            var_type: param_type,
                        });
                    }
                }

                // Extract return type
                let return_type = match &func.sig.output {
                    syn::ReturnType::Type(_, ty) => extract_type(ty),
                    _ => Type::Int32,
                };

                // Extract function body
                let mut body = Vec::new();
                for stmt in func.block.stmts {
                    let ir_stmt = extract_statement(&stmt)?;
                    body.push(ir_stmt);
                }

                functions.push(Function {
                    name,
                    parameters,
                    return_type,
                    body,
                });
            }
            _ => {}
        }
    }

    Ok(Program { functions })
}

// Helper functions omitted for brevity...
```

---

## Appendix C: References

### Internal Documentation
- PR #1060: zkLean extractor implementation
- [zklean-extractor/](../../zklean-extractor/): Source code for extraction tool
- [jolt-core/src/zkvm/](../../jolt-core/src/zkvm/): Verifier implementation

### Industry Approaches
- **RISC Zero**: [groth16_proof](https://github.com/risc0/risc0/tree/main/groth16_proof) - Manual Circom approach
  - [stark_verify.circom](https://github.com/risc0/risc0/blob/main/groth16_proof/groth16/stark_verify.circom) - 55.7MB circuit
  - [Trusted Setup Ceremony](https://dev.risczero.com/api/trusted-setup-ceremony)
- **SP1**: [recursion/gnark-ffi](https://github.com/succinctlabs/sp1/tree/dev/crates/recursion/gnark-ffi) - DSL compiler approach
  - [SP1 Testnet Launch](https://blog.succinct.xyz/sp1-testnet/) - Recursion compiler details
  - [sp1-recursion-gnark-ffi](https://crates.io/crates/sp1-recursion-gnark-ffi) - FFI crate

### Target Framework
- [Gnark documentation](https://docs.gnark.consensys.io/): Target circuit framework
- [Gnark benchmarks](https://blog.celer.network/2023/08/04/the-pantheon-of-zero-knowledge-proof-development-frameworks/): 5-10× faster than Arkworks

### Research on Automatic Approaches
- zkLLVM: PLONK output, not R1CS - incompatible with Groth16
- No existing tool for automatic Rust → R1CS conversion

---

**Document Status**: Educational guide + proposal
**Date**: 2025-11-10 (Updated 2025-11-25 with industry research findings)
**Next Steps**: Validate PoC for single sumcheck round
