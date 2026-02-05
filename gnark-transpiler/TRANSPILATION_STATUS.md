# Jolt Verifier → Gnark: Documentación Técnica de Transpilación

## Tabla de Contenidos

1. [Transpilación por ejecución simbólica](#1-transpilación-por-ejecución-simbólica)
2. [Arquitectura del sistema](#2-arquitectura-del-sistema)
3. [Implementación Rust: Ejecución simbólica](#3-implementación-rust-ejecución-simbólica)
4. [Generación de código Go](#4-generación-de-código-go)
5. [Trabajo completado](#5-trabajo-completado)
6. [Problema actual: Stage 2](#6-problema-actual-stage-2)
7. [Trabajo pendiente](#7-trabajo-pendiente)
8. [Procedimiento para continuar](#8-procedimiento-para-continuar)

---

## 1. Transpilación por ejecución simbólica

### Por qué ejecución simbólica

Existen varias formas de convertir un verificador Rust a un circuito Gnark:

| Técnica | Descripción | Problema |
|---------|-------------|----------|
| **Reescritura manual** | Traducir el código Rust a Go línea por línea | Propenso a errores, difícil de mantener |
| **Compilación directa** | Compilar Rust → WASM → circuito | WASM tiene overhead enorme (memoria, control flow) |
| **Análisis estático** | Parsear el AST de Rust y generar código Go | No maneja bien código genérico, macros, traits |
| **Ejecución simbólica** | Ejecutar el código con valores simbólicos | ✅ Reutiliza el código existente exactamente |

**Ventajas de ejecución simbólica:**

1. **Correctitud por construcción**: El circuito hace exactamente lo que hace el verificador, porque ES el verificador ejecutándose
2. **Mantenibilidad**: Si jolt-core cambia, solo hay que re-ejecutar la transpilación
3. **Generics de Rust**: El verificador ya es genérico sobre `F: JoltField`, solo hay que instanciarlo con `F = MleAst`
4. **Sin parsing**: No necesitamos entender el código, solo ejecutarlo

**Desventaja:**

- Solo captura UN camino de ejecución (el del proof concreto usado). Si hay branches que dependen de valores del proof, solo se transpila el branch tomado. En la práctica esto no es problema porque el verificador de Jolt es determinístico dado el tamaño del trace.

### Método: Ejecución simbólica con Abstract Syntax Trees

En lugar de ejecutar el verificador con valores concretos, ejecutamos con valores simbólicos.

El tipo `MleAst` representa expresiones simbólicas que incluyen:
- Variables de entrada (elementos del proof)
- Constantes (elementos de campo)
- Operaciones aritméticas de campo: +, -, ×, ÷, neg, inv
- Hash Poseidon: `Poseidon(MleAst, MleAst) → MleAst`
- Operaciones de bytes: `ToBytes(MleAst) → [MleAst; 32]`, `FromBytes([MleAst; 32]) → MleAst`

Cada operación en el verificador construye un nodo en el AST en lugar de computar un valor.

**Ejemplo:**

```
Ejecución concreta:          Ejecución simbólica:
─────────────────           ────────────────────
a = 5                       a = Var(0)
b = 3                       b = Var(1)
c = a + b  → 8              c = Add(Var(0), Var(1))
d = c * 2  → 16             d = Mul(Add(Var(0), Var(1)), Const(2))
```

### Captura de constraints

El verificador de Jolt realiza comparaciones de igualdad (assertions). Durante la ejecución simbólica, cada comparación `a == b` se captura como un par `(a, b)`.

Los pares capturados se convierten en assertions del circuito Gnark: `api.AssertIsEqual(a, b)`.

### Conversión a R1CS

Gnark convierte las expresiones aritméticas a R1CS (Rank-1 Constraint System). Para una expresión como:
```
(a + b) × c - d = 0
```

Se introducen variables intermedias:
```
t₁ = a + b       (lineal, no requiere constraint multiplicativo)
t₁ × c = t₂      (constraint R1CS: t₁ · c = t₂)
t₂ - d = 0       (lineal, se absorbe)
```

El número de constraints R1CS es aproximadamente igual al número de multiplicaciones en las expresiones.

### Eager vs Lazy transpilación

Usamos transpilación **eager** (todo se expande durante la ejecución):

```
EAGER (lo que hacemos):
─────────────────────
verify() ejecuta → cada operación crea nodo inmediatamente
                 → al final tenemos grafo completo
                 → generamos código Go de una vez

LAZY (alternativa):
───────────────────
verify() ejecuta → operaciones crean "thunks" (computaciones diferidas)
                 → al final tenemos grafo de dependencias
                 → evaluamos solo lo necesario para las assertions
```

**Por qué eager**: El verificador de Jolt ya está diseñado para ser eficiente. No hay computaciones "muertas" que se descarten. Lazy agregaría complejidad sin beneficio.

---

## 2. Arquitectura del sistema

### Diagrama de componentes

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            RUST SIDE                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐ │
│  │ Real Proof       │     │ symbolize_proof  │     │ Symbolic Proof   │ │
│  │ (concrete values)│────▶│                  │────▶│ (MleAst vars)    │ │
│  │ (concrete values)│     │ VarAllocator     │     │ (AST variables)  │ │
│  └──────────────────┘     └──────────────────┘     └────────┬─────────┘ │
│                                                              │          │
│                                                              ▼          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    TranspilableVerifier                           │  │
│  │  ┌─────────────────────────────────────────────────────────────┐ │  │
│  │  │ Generics:                                                    │ │  │
│  │  │   F = MleAst                    (symbolic field)             │ │  │
│  │  │   PCS = AstCommitmentScheme     (dummy, no curve ops)        │ │  │
│  │  │   T = PoseidonAstTranscript     (symbolic transcript)        │ │  │
│  │  │   A = MleOpeningAccumulator     (symbolic accumulator)       │ │  │
│  │  └─────────────────────────────────────────────────────────────┘ │  │
│  │                                                                   │  │
│  │                                                                   │  │
│  │  ┌─────────────────────────────────────────────────────────────┐ │  │
│  │  │ CONSTRAINT_MODE = true (activo durante toda la verificación)│ │  │
│  │  │ Cada MleAst::eq(a,b) → ASSERTIONS.push(a - b)               │ │  │
│  │  └─────────────────────────────────────────────────────────────┘ │  │
│  │                                                                   │  │
│  │  verify_stage1() ──▶ BatchedSumcheck::verify() ──▶ assertions   │  │
│  │  verify_stage2() ──▶ BatchedSumcheck::verify() ──▶ assertions   │  │
│  │  ...                                                              │  │
│  │  verify_stage6() ──▶ BatchedSumcheck::verify() ──▶ assertions   │  │
│  │                                                                   │  │
│  │  Resultado: Vec<MleAst> expresiones que deben ser = 0            │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                    │                                    │
│                                    ▼                                    │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                       MemoizedCodeGen                             │  │
│  │                                                                   │  │
│  │  1. count_refs(ast) - cuenta referencias a cada nodo             │  │
│  │  2. Para nodos con refs > 1: genera variable CSE                 │  │
│  │  3. generate_expr(ast) - genera código Go recursivamente         │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                    │                                    │
└────────────────────────────────────┼────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                              GO SIDE                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    Generated Circuit                              │  │
│  │                                                                   │  │
│  │  type JoltStages16Circuit struct {                               │  │
│  │      X0, X1, ..., Xn frontend.Variable `gnark:",public"`         │  │
│  │  }                                                                │  │
│  │                                                                   │  │
│  │  func (c *JoltStages16Circuit) Define(api frontend.API) error {  │  │
│  │      // CSE bindings                                              │  │
│  │      cse_0 := api.Add(c.X0, c.X1)                                │  │
│  │      cse_1 := api.Mul(cse_0, c.X2)                               │  │
│  │      ...                                                          │  │
│  │      // Constraints                                               │  │
│  │      a0 := <expr>                                                 │  │
│  │      api.AssertIsEqual(a0, 0)                                    │  │
│  │      ...                                                          │  │
│  │      return nil                                                   │  │
│  │  }                                                                │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                    │                                    │
│                                    ▼                                    │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    Gnark Compilation                              │  │
│  │                                                                   │  │
│  │  frontend.Compile() ──▶ R1CS constraint system                   │  │
│  │  groth16.Setup()    ──▶ (ProvingKey, VerifyingKey)              │  │
│  │  groth16.Prove()    ──▶ Groth16 proof                           │  │
│  │  groth16.Verify()   ──▶ bool                                     │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Flujo de datos concreto

La transpilación produce dos artefactos que Go consume:

```
RUST SIDE                                    GO SIDE
─────────                                    ───────

1. Código Go generado (stages16_circuit.go)
   ┌────────────────────────────────┐
   │ type JoltStages16Circuit {     │
   │   X0, X1, ... frontend.Variable│  ←── Variables del proof
   │ }                              │
   │                                │
   │ func Define(api) {             │
   │   cse_0 := api.Add(...)        │  ←── Expresiones del AST
   │   api.AssertIsEqual(...)       │  ←── Assertions capturadas
   │ }                              │
   └────────────────────────────────┘

2. Bundle JSON (stages16_bundle.json)
   ┌────────────────────────────────┐
   │ {                              │
   │   "variables": {               │
   │     "X0": "123456...",         │  ←── Valores concretos del proof
   │     "X1": "789012...",         │      (witness para el circuito)
   │     ...                        │
   │   },                           │
   │   "num_vars": 847              │
   │ }                              │
   └────────────────────────────────┘
```

**El bundle JSON contiene:**

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `variables` | `Map<String, String>` | Nombre de variable → valor decimal (string porque son 256 bits) |
| `num_vars` | `u32` | Número total de variables de entrada |

**Cómo Go usa estos archivos:**

```go
// 1. Cargar bundle
bundle := loadBundle("stages16_bundle.json")

// 2. Crear witness (assignment)
assignment := JoltStages16Circuit{}
assignment.X0 = bundle.variables["X0"]
assignment.X1 = bundle.variables["X1"]
// ... (generado automáticamente)

// 3. Crear witness completo
witness, _ := frontend.NewWitness(&assignment, ecc.BN254.ScalarField())

// 4. Compilar circuito (genera R1CS)
cs, _ := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, &JoltStages16Circuit{})

// 5. Setup Groth16 (genera proving key y verifying key)
pk, vk, _ := groth16.Setup(cs)

// 6. Generar proof
proof, _ := groth16.Prove(cs, pk, witness)

// 7. Verificar (esto es lo que iría on-chain)
groth16.Verify(proof, vk, publicWitness)
```

### Tipos principales y sus roles

#### MleAst (Multi-Linear Extension AST)

Ubicación: `zklean-extractor/src/mle_ast.rs`

```rust
pub struct MleAst {
    root: NodeId,           // Índice en el arena global
    reg_name: Option<u8>,   // Metadata opcional
}

// Arena global (append-only, thread-safe)
static NODES: RwLock<Vec<Node>> = ...;

pub enum Node {
    Atom(Atom),
    Add(Edge, Edge),
    Sub(Edge, Edge),
    Mul(Edge, Edge),
    Div(Edge, Edge),
    Neg(Edge),
    Inv(Edge),
    Poseidon(Edge, Edge, Edge),
    Keccak256(Edge),
    ByteReverse(Edge),
    Truncate128(Edge),
    Truncate128Reverse(Edge),
    MulTwoPow192(Edge),
}

pub enum Atom {
    Scalar([u64; 4]),    // Elemento de F (256 bits en limbs de 64 bits)
    Var(u16),            // Variable simbólica indexada
    NamedVar(u16),       // Variable con nombre descriptivo
}

pub enum Edge {
    Atom(Atom),          // Valor directo (evita crear nodo)
    NodeRef(NodeId),     // Referencia a nodo existente
}
```

**Invariantes:**
- Los nodos nunca se modifican después de crearse (append-only)
- NodeId es un índice estable en el arena
- Edge::Atom optimiza casos donde no se necesita crear nodo (constantes pequeñas, variables)

**Implementación de operaciones:**

```rust
impl std::ops::Mul<&Self> for MleAst {
    fn mul(mut self, rhs: &Self) -> Self::Output {
        // Optimización: x * 0 = 0, 0 * x = 0
        if self.is_zero() || rhs.is_zero() {
            return Self::zero();
        }
        // Crear nodo Mul en el arena
        let lhs_edge = edge_for_root(self.root);
        let rhs_edge = edge_for_root(rhs.root);
        self.root = insert_node(Node::Mul(lhs_edge, rhs_edge));
        self
    }
}
```

#### MleOpeningAccumulator

Ubicación: `gnark-transpiler/src/mle_opening_accumulator.rs`

El verificador de Jolt usa un `OpeningAccumulator` para:
1. Almacenar claims de evaluación de polinomios
2. Acumular puntos de apertura durante la verificación
3. Batch las aperturas para verificación final

```rust
pub struct MleOpeningAccumulator {
    pub openings: BTreeMap<OpeningId, (Vec<MleAst>, MleAst)>,
    //                     ─────────   ──────────  ──────
    //                     Clave       Punto       Claim
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum OpeningId {
    Virtual(VirtualPolynomial, SumcheckId),
    Committed(CommittedPolynomial, SumcheckId),
    UntrustedAdvice(SumcheckId),
    TrustedAdvice(SumcheckId),
}
```

El trait `OpeningAccumulator<F>` define la interfaz:

```rust
pub trait OpeningAccumulator<F: JoltField> {
    // Lectura: obtener claim y punto de apertura
    fn get_virtual_polynomial_opening(
        &self,
        polynomial: VirtualPolynomial,
        sumcheck: SumcheckId,
    ) -> (OpeningPoint<BIG_ENDIAN, F>, F);

    // Escritura: registrar punto de apertura derivado del sumcheck
    fn append_virtual<T: Transcript>(
        &mut self,
        transcript: &mut T,
        polynomial: VirtualPolynomial,
        sumcheck: SumcheckId,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    );

    // ... más métodos para committed, advice, dense, sparse
}
```

#### PoseidonAstTranscript

Ubicación: `zklean-extractor/src/poseidon_ast_transcript.rs`

El transcript Fiat-Shamir genera challenges "aleatorios" derivados del estado actual.
En ejecución simbólica, los challenges son expresiones MleAst que representan la evaluación de Poseidon.

```rust
pub struct PoseidonAstTranscript {
    state: MleAst,      // Estado actual del transcript
    n_rounds: u32,      // Contador de rondas Poseidon
}

impl Transcript for PoseidonAstTranscript {
    type Challenge = MleAst;

    fn append_scalar(&mut self, scalar: &MleAst) {
        // state = Poseidon(state, n_rounds, scalar)
        self.state = MleAst::poseidon(&self.state, &MleAst::from_u64(self.n_rounds), scalar);
        self.n_rounds += 1;
    }

    fn challenge(&mut self) -> MleAst {
        // El challenge es el estado actual
        // Luego actualizamos el estado para el próximo challenge
        let challenge = self.state.clone();
        self.state = MleAst::poseidon(&self.state, &MleAst::from_u64(self.n_rounds), &MleAst::zero());
        self.n_rounds += 1;
        challenge
    }
}
```

#### AstCommitmentScheme

Ubicación: `gnark-transpiler/src/ast_commitment_scheme.rs`

Dummy implementation del trait `CommitmentScheme`. Los commitments reales son puntos de curva elíptica que no podemos representar simbólicamente (requieren operaciones de grupo, no de campo).

```rust
pub struct AstCommitmentScheme;

impl CommitmentScheme for AstCommitmentScheme {
    type Field = MleAst;
    type Commitment = ();              // No almacenamos nada
    type Proof = ();
    type VerifierSetup = AstVerifierSetup;

    // Los métodos que requieren curva elíptica hacen panic o retornan dummy
}
```

---

## 3. Implementación Rust: Ejecución simbólica

### 3.1 MleAst: El tipo simbólico

`MleAst` es el tipo central que representa expresiones simbólicas sobre el campo escalar de BN254.

#### Por qué una arena global

El grafo de expresiones puede tener millones de nodos con sharing extensivo (un nodo puede ser referenciado por muchos otros). Las alternativas:

| Estructura | Problema |
|------------|----------|
| `Box<Node>` | No hay sharing, cada uso clona todo el subárbol |
| `Rc<Node>` | Sharing funciona, pero reference counting tiene overhead |
| `Arc<Node>` | Peor que Rc por el atomic ref counting |
| **Arena global** | ✅ NodeId es un `usize`, copiar es O(1), sin overhead de ref counting |

**Trade-off**: Los nodos nunca se liberan durante la ejecución. Esto es aceptable porque:
1. La transpilación es un proceso batch (no un servidor long-running)
2. El grafo final se necesita completo para generar código
3. La memoria usada (~100MB para Stage 1) es manejable

**Implementación thread-safe**:

```rust
static NODE_ARENA: OnceLock<RwLock<Vec<Node>>> = OnceLock::new();

// Insertar es write lock (exclusivo)
pub fn insert_node(node: Node) -> NodeId {
    let mut guard = arena.write().expect("poisoned");
    let id = guard.len();
    guard.push(node);
    id
}

// Leer es read lock (compartido, múltiples lectores)
pub fn get_node(id: NodeId) -> Node {
    let guard = arena.read().expect("poisoned");
    guard[id]  // Node es Copy
}
```

**Ubicación:** `zklean-extractor/src/mle_ast.rs`

#### Estructura de datos

Los nodos del AST se almacenan en una arena global thread-safe:

```rust
type NodeId = usize;
type Scalar = [u64; 4];  // 256-bit en little-endian

static NODE_ARENA: OnceLock<RwLock<Vec<Node>>> = OnceLock::new();

pub fn insert_node(node: Node) -> NodeId {
    let arena = node_arena();
    let mut guard = arena.write().expect("node arena poisoned");
    let id = guard.len();
    guard.push(node);
    id
}

pub fn get_node(id: NodeId) -> Node {
    let arena = node_arena();
    let guard = arena.read().expect("node arena poisoned");
    guard.get(id).copied().expect("invalid node reference")
}
```

Esto permite que `MleAst` sea `Copy` (solo contiene un `NodeId`) mientras el grafo crece sin límite.

#### Átomos

```rust
pub enum Atom {
    /// Constante de campo: [u64; 4] en little-endian
    /// Value = limb0 + limb1*2^64 + limb2*2^128 + limb3*2^192
    Scalar(Scalar),

    /// Variable de entrada del circuito (índice en el struct de inputs)
    Var(Index),

    /// Variable let-bound para CSE (común sub-expresión eliminada)
    NamedVar(LetBinderIndex),
}
```

#### Edges y Nodes

Un `Edge` puede ser un átomo (inlined) o una referencia a un nodo complejo:

```rust
pub enum Edge {
    Atom(Atom),
    NodeRef(NodeId),
}

pub enum Node {
    Atom(Atom),

    // Operaciones aritméticas de campo
    Neg(Edge),
    Inv(Edge),
    Add(Edge, Edge),
    Sub(Edge, Edge),
    Mul(Edge, Edge),
    Div(Edge, Edge),

    // Hash Poseidon (3 inputs: state, n_rounds, data)
    Poseidon(Edge, Edge, Edge),

    // Keccak256 de un field element
    Keccak256(Edge),

    // Transformaciones de bytes
    ByteReverse(Edge),           // LE bytes → reverse → from_le_bytes_mod_order
    Truncate128Reverse(Edge),    // Truncar a 128 bits, reverse, shift ×2^128
    Truncate128(Edge),           // Truncar a 128 bits, reverse (sin shift)
    MulTwoPow192(Edge),          // Multiplicar por 2^192 (BE-padding para append_u64)
}
```

#### El struct MleAst

```rust
pub struct MleAst {
    root: NodeId,           // Índice del nodo raíz en la arena
    reg_name: Option<char>, // Nombre del registro (para debug)
}
```

`MleAst` implementa:
- `JoltField` trait: permite usarlo donde jolt-core espera elementos de campo
- Operaciones aritméticas (`Add`, `Sub`, `Mul`, `Div`, `Neg`)
- `PartialEq`: en modo constraint, captura assertions en lugar de comparar

#### Montgomery form: cuidados en la interfaz Rust ↔ Go

**Montgomery form** es una representación donde el valor `a` se almacena como `a * R mod p`, donde `R = 2^256 mod p`. Esto permite multiplicaciones modulares más eficientes. El problema es que diferentes funciones de ark-ff asumen diferentes representaciones, y al replicar operaciones de Rust en Go (hints), hay que saber exactamente qué hace cada función.

**Constante importante**:
```
R⁻¹ mod p = 9915499612839321149637521777990102151350674507940716049588462388200839649614
```

---

**Funciones de ark-ff y sus semánticas**:

| Función | Input | Output | Qué hace internamente |
|---------|-------|--------|----------------------|
| `from_le_bytes_mod_order(bytes)` | bytes LE | Fr | Interpreta bytes como valor directo, luego convierte a Montgomery |
| `from_bigint_unchecked(bigint)` | BigInt | Fr | **Asume input ya en Montgomery**, multiplica por R⁻¹ |
| `into_bigint()` | Fr | BigInt | Convierte de Montgomery a valor directo |

---

**Caso 1: MleAst (lado Rust)**

`MleAst` almacena constantes en forma directa (no Montgomery):

```rust
const SCALAR_ZERO: Scalar = [0, 0, 0, 0];  // Valor lógico 0
const SCALAR_ONE: Scalar = [1, 0, 0, 0];   // Valor lógico 1
```

Al convertir `Fr` → `MleAst`, hay que usar `into_bigint()`:

```rust
// INCORRECTO: fr.0.0 contiene valor en Montgomery form
// CORRECTO:
let bigint = fr.into_bigint();  // Montgomery → directo
```

---

**Caso 2: Hints de Go (lado Gnark)**

Los hints deben replicar exactamente el comportamiento de Rust. La diferencia crítica:

**`challenge_scalar`** (usado en r0, batching_coeff, sumcheck challenges):
```rust
// Rust:
let bytes = hash.finalize()[0..16];
let reversed = bytes.iter().rev().collect();
Fr::from_le_bytes_mod_order(&reversed)  // ← NO pasa por R⁻¹
```

```go
// Go (truncate128Hint):
// Solo reverse bytes e interpretar como LE
// NO multiplicar por R⁻¹
```

**`challenge_scalar_optimized`** (usado en algunos challenges):
```rust
// Rust:
let mont_challenge = MontU128Challenge::from_bytes(bytes);
let fr: Fr = mont_challenge.into();  // Llama from_bigint_unchecked
// from_bigint_unchecked multiplica por R⁻¹ internamente
```

```go
// Go (truncate128ReverseHint):
// Construir bigint = low * 2^128 + high * 2^192
// MULTIPLICAR por R⁻¹ mod p
result := new(big.Int).Mul(bigintValue, bn254RInv)
result.Mod(result, bn254FrModulus)
```

---

**Resumen de hints**:

| Hint | Operación Rust | Usa R⁻¹? |
|------|---------------|----------|
| `truncate128Hint` | `from_le_bytes_mod_order` | NO |
| `truncate128ReverseHint` | `from_bigint_unchecked` | SÍ |
| `byteReverseHint` | `from_le_bytes_mod_order` | NO |
| `appendU64TransformHint` | BE pack + LE interpret | NO |

**Regla general**: Si la función Rust termina en `from_bigint_unchecked`, el hint Go necesita multiplicar por R⁻¹. Si termina en `from_le_bytes_mod_order`, no.

#### Modo constraint

Thread-locals para capturar assertions durante ejecución simbólica:

```rust
thread_local! {
    static SYMBOLIC_CONSTRAINTS: RefCell<Vec<MleAst>> = RefCell::new(Vec::new());
    static CONSTRAINT_MODE: RefCell<bool> = RefCell::new(false);
}

pub fn enable_constraint_mode() { ... }
pub fn disable_constraint_mode() { ... }
pub fn take_constraints() -> Vec<MleAst> { ... }

fn add_constraint(constraint: MleAst) {
    SYMBOLIC_CONSTRAINTS.with(|cell| {
        cell.borrow_mut().push(constraint);
    });
}
```

Cuando `CONSTRAINT_MODE = true`, la implementación de `PartialEq` hace:

```rust
impl PartialEq for MleAst {
    fn eq(&self, other: &Self) -> bool {
        if is_constraint_mode() {
            // Capturar constraint: self - other debe ser 0
            add_constraint(*self - *other);
            return true;  // Retornar true para que verify() continúe
        }
        // Comparación normal por NodeId
        self.root == other.root
    }
}
```

#### Comunicación con el transcript

El trait `Transcript` de jolt-core tiene métodos como:
- `append_scalar<F>(&mut self, scalar: &F)` - llama `F::serialize`
- `challenge_scalar<F>(&mut self) -> F` - llama `F::from_bytes`

Pero `MleAst::serialize` y `MleAst::from_bytes` no tienen sentido semántico para ASTs.
Se usan thread-locals para "tunnelear" los valores MleAst a través de estas interfaces:

```rust
thread_local! {
    static PENDING_APPEND: RefCell<Option<MleAst>> = RefCell::new(None);
    static PENDING_CHALLENGE: RefCell<Option<MleAst>> = RefCell::new(None);
}

// En MleAst::serialize_with_mode:
pub fn serialize_with_mode<W: Write>(&self, ...) -> Result<(), SerializationError> {
    set_pending_append(*self);  // Guardar el MleAst real
    Ok(())
}

// PoseidonAstTranscript::append_scalar llama serialize, luego:
pub fn append_scalar<F: JoltField>(&mut self, _scalar: &F) {
    let ast = take_pending_append().expect("MleAst must set pending");
    // Ahora tenemos el MleAst real, podemos construir nodos Poseidon
}
```

### 3.2 PoseidonAstTranscript: Fiat-Shamir simbólico

**Ubicación:** `gnark-transpiler/src/poseidon_transcript.rs`

El transcript simbólico construye nodos `Poseidon` en lugar de computar hashes:

```rust
pub struct PoseidonAstTranscript {
    /// Estado interno del transcript (simbólico)
    state: MleAst,

    /// Contador de operaciones (para constantes n_rounds)
    operation_count: usize,
}
```

Cuando se llama `challenge_scalar`:

```rust
fn challenge_scalar<F: JoltField>(&mut self) -> F {
    // 1. Computar hash simbólico
    let n_rounds = MleAst::from_u64(self.operation_count as u64);
    let hash_output = MleAst::poseidon(&self.state, &n_rounds, &MleAst::zero());

    // 2. Aplicar transformaciones de bytes (matching el transcript real)
    let truncated = MleAst::truncate_128(&hash_output);

    // 3. Actualizar estado
    self.state = hash_output;
    self.operation_count += 1;

    // 4. Retornar vía thread-local
    set_pending_challenge(truncated);
    F::from_bytes(&[0u8; 32])  // El valor real viene del thread-local
}
```

#### Mapeo Poseidon Rust → Go

El hash Poseidon tiene parámetros configurables. Ambos lados deben usar exactamente los mismos:

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| Field | BN254 Fr | Campo escalar de la curva |
| Width | 3 | Número de elementos en el estado |
| Full rounds | 8 | Rondas con S-box en todos los elementos |
| Partial rounds | 57 | Rondas con S-box solo en un elemento |
| S-box | x⁵ | Función no lineal |

**En Rust** (`jolt-core/src/utils/transcript.rs`):
```rust
use ark_crypto_primitives::sponge::poseidon::PoseidonConfig;
// Usa parámetros de ark-crypto-primitives para BN254
```

**En Go** (`gnark-transpiler/go/poseidon/poseidon.go`):
```go
import "github.com/consensys/gnark-crypto/ecc/bn254/fr/poseidon"
// Usa implementación de gnark-crypto con mismos parámetros
```

**Verificación de compatibilidad**: El test `TestPoseidonCompatibility` en Go compara outputs con valores conocidos de Rust.

#### Thread-local tunneling en detalle

El problema: `Transcript::append_scalar<F>` llama `F::serialize()`, pero no podemos serializar un AST a bytes de forma que preserve su estructura.

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Flujo normal (F = Fr):                                                   │
│                                                                          │
│ transcript.append_scalar(&fr_value)                                     │
│     └──▶ fr_value.serialize(&mut bytes)  // Fr → [u8; 32]              │
│         └──▶ transcript absorbe bytes                                   │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ Flujo simbólico (F = MleAst):                                           │
│                                                                          │
│ transcript.append_scalar(&mle_ast)                                      │
│     │                                                                    │
│     │  [1] MleAst::serialize() guarda en thread-local                   │
│     │      PENDING_APPEND.set(Some(mle_ast))                            │
│     │                                                                    │
│     └──▶ PoseidonAstTranscript::append_scalar()                         │
│             │                                                            │
│             │  [2] Recupera el MleAst del thread-local                   │
│             │      let ast = PENDING_APPEND.take()                       │
│             │                                                            │
│             └──▶ self.state = Poseidon(self.state, n_rounds, ast)       │
└─────────────────────────────────────────────────────────────────────────┘
```

Este patrón es necesario porque el trait `Transcript` no es genérico sobre el tipo de campo en su interfaz (usa serialización).

### 3.3 MleOpeningAccumulator: Estado de openings

**Ubicación:** `gnark-transpiler/src/mle_opening_accumulator.rs`

Durante la verificación, el verificador lee claims del proof y deriva puntos de evaluación:

```rust
pub struct MleOpeningAccumulator {
    /// Map: OpeningId → (punto de evaluación, claim)
    pub openings: BTreeMap<OpeningId, (Vec<MleAst>, MleAst)>,
}
```

#### Estructura de OpeningId

Los openings se identifican por qué polinomio y en qué sumcheck:

```rust
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum OpeningId {
    /// Polinomio virtual (combinación de polinomios base)
    Virtual(VirtualPolynomial, SumcheckId),

    /// Polinomio committed (tiene commitment en el proof)
    Committed(CommittedPolynomial, SumcheckId),

    /// Advice no verificado (viene del prover, se verifica después)
    UntrustedAdvice(SumcheckId),

    /// Advice verificado (ya se verificó en stage anterior)
    TrustedAdvice(SumcheckId),
}

/// Identifica un sumcheck específico
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum SumcheckId {
    SpartanOuter,           // Stage 1
    ProductVirtual,         // Stage 2
    RamRafEvaluation,       // Stage 2
    RamReadWriteChecking,   // Stage 2
    OutputCheck,            // Stage 2
    // ... más sumchecks para stages 3-6
}

/// Polinomios virtuales (no tienen commitment propio)
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum VirtualPolynomial {
    Product,                // ∏ᵢ (1 + rᵢ·(fᵢ - 1))
    WriteLookupOutputToRD,
    WritePCtoRD,
    ShouldBranch,
    ShouldJump,
    // ... más
}
```

#### De dónde vienen los claims iniciales

El accumulator se inicializa con claims del proof **antes** de ejecutar la verificación:

```rust
// En transpile_stages.rs:

// 1. Cargar proof real (tiene claims concretos)
let proof: JoltProof<Fr, HyperKZG, PoseidonTranscript> = load_proof("proof.bin");

// 2. Extraer opening_ids y sus claims del proof
let opening_ids: Vec<(OpeningId, Fr)> = proof.extract_opening_claims();

// 3. Crear accumulator simbólico
let mut accumulator = MleOpeningAccumulator::new();

// 4. Para cada claim, crear una variable simbólica
let mut var_idx = 0;
for (id, _concrete_claim) in opening_ids {
    let symbolic_claim = MleAst::from_var(var_idx);
    var_idx += 1;

    // Punto inicial vacío (se llenará durante verificación)
    let empty_point: Vec<MleAst> = vec![];

    accumulator.openings.insert(id, (empty_point, symbolic_claim));
}
```

El claim es **simbólico** (una variable de entrada del circuito). Durante la verificación, el sumcheck deriva el **punto de evaluación** (una expresión simbólica de challenges). Al final:

- **Claim**: Variable de entrada (viene del proof)
- **Punto**: Expresión simbólica (derivada de challenges del transcript)

La verificación final (stages 7-8) comprueba que evaluar el polinomio en el punto da el claim.

#### Implementación del trait

```rust
impl OpeningAccumulator<MleAst> for MleOpeningAccumulator {
    fn get_virtual_polynomial_opening(
        &self,
        polynomial: VirtualPolynomial,
        sumcheck: SumcheckId,
    ) -> (OpeningPoint<BIG_ENDIAN, MleAst>, MleAst) {
        let key = OpeningId::Virtual(polynomial, sumcheck);
        let (point, claim) = self.openings.get(&key).unwrap();
        (OpeningPoint::new(point.clone()), claim.clone())
    }

    fn append_virtual<T: Transcript>(
        &mut self,
        _transcript: &mut T,
        polynomial: VirtualPolynomial,
        sumcheck: SumcheckId,
        opening_point: OpeningPoint<BIG_ENDIAN, MleAst>,
    ) {
        let key = OpeningId::Virtual(polynomial, sumcheck);
        if let Some((stored_point, _)) = self.openings.get_mut(&key) {
            *stored_point = opening_point.r;
        }
    }
}
```

### 3.4 TranspilableVerifier: El verificador genérico

**Ubicación:** `jolt-core/src/zkvm/transpilable_verifier.rs`

El verificador está parametrizado sobre:

```rust
pub struct TranspilableVerifier<
    'a,
    F: JoltField,
    PCS: CommitmentScheme<F>,
    T: Transcript,
    A: OpeningAccumulator<F>,
> {
    preprocessing: &'a JoltVerifierPreprocessing<F, PCS>,
    proof: JoltProof<F, PCS, T>,
    transcript: T,
    opening_accumulator: A,
    // ...
}
```

Para transpilación, se instancia con:
- `F = MleAst`
- `PCS = AstCommitmentScheme` (dummy, no hace operaciones de curva)
- `T = PoseidonAstTranscript`
- `A = MleOpeningAccumulator`

### 3.5 Flujo de ejecución simbólica

```
1. Cargar proof real:
   proof_bytes → RV64IMACProof (valores concretos Fr)

2. Simbolizar proof:
   RV64IMACProof → JoltProof<MleAst, ...>
   - Cada campo escalar → MleAst::from_var(idx)
   - VarAllocator asigna índices únicos

3. Crear accumulator con claims simbólicos:
   MleOpeningAccumulator::from_opening_ids(proof.opening_ids(), start_idx)

4. Crear verificador:
   TranspilableVerifier::new_with_accumulator(...)

5. Habilitar modo constraint:
   enable_constraint_mode()

6. Ejecutar verificación:
   verifier.verify()
   - Todas las operaciones (+, -, *, /) construyen nodos MleAst
   - Poseidon construye nodos Node::Poseidon
   - Comparaciones (==) capturan assertions

7. Recolectar assertions:
   let assertions: Vec<(MleAst, MleAst)> = take_assertions();
```

### 3.6 Estructura del verificador por stages

El verificador ejecuta stages 1-6. Cada stage tiene una estructura similar:

1. **Uni-skip first round** (stages 1, 2): Optimización donde la primera ronda del sumcheck se verifica con estructura especial (el polinomio tiene forma conocida en la primera variable)

2. **Batched sumcheck verification**: Verifica múltiples instancias de sumcheck en paralelo usando random linear combination

3. **Cache openings**: Registra los puntos derivados (de challenges) en el accumulator para verificación posterior

#### Detalle de cada stage

| Stage | Sumchecks | Rondas | Qué verifica |
|-------|-----------|--------|--------------|
| **1** | 1 (Spartan outer) | 16 | R1CS satisfiability via Spartan |
| **2** | 5 (batched) | 16 | Product virtualization + RAM/Registers initial checks |
| **3** | 4 (batched) | 16 | RAM read/write permutation + Registers permutation |
| **4** | 3 (batched) | 16 | Instruction lookups RA/RAF + Bytecode RA/RAF |
| **5** | 2 (batched) | 16 | Hamming weight + Booleanity checks |
| **6** | 1 | 16 | Opening reduction sumcheck |

**Stage 1: Spartan Outer**

```rust
fn verify_stage1(&mut self) -> Result<(), anyhow::Error> {
    // Uni-skip: primera ronda tiene estructura especial
    // Verifica: ∑ᵢ eq(τ, i) · (Az(i) · Bz(i) - Cz(i)) = 0
    // Donde Az, Bz, Cz son las matrices R1CS evaluadas

    let uni_skip_verifier = OuterUniSkipVerifier::new(...);
    let remainder_verifier = OuterRemainingSumcheckVerifier::new(...);

    BatchedSumcheck::verify(&[uni_skip_verifier, remainder_verifier], ...)?;

    // Claims resultantes: evaluaciones de Az, Bz, Cz en punto aleatorio
}
```

**Stage 2: Product + RAM/Registers**

```rust
fn verify_stage2(&mut self) -> Result<(), anyhow::Error> {
    // 5 sumchecks batched con RLC (Random Linear Combination)

    let verifiers: Vec<Box<dyn SumcheckInstanceVerifier<...>>> = vec![
        // 1. Product virtualization
        Box::new(ProductVirtualRemainderVerifier::new(...)),

        // 2. RAM RAF evaluation
        Box::new(RamRafEvaluationSumcheckVerifier::new(...)),

        // 3. RAM read/write checking ← PROBLEMA: usa EqPolynomial::evals
        Box::new(RamReadWriteCheckingVerifier::new(...)),

        // 4. Output check
        Box::new(OutputSumcheckVerifier::new(...)),

        // 5. Instruction lookups claim reduction
        Box::new(InstructionLookupsClaimReductionSumcheckVerifier::new(...)),
    ];

    BatchedSumcheck::verify(&verifiers, ...)?;
}
```

**Stages 3-6** siguen el mismo patrón con diferentes verifiers.

#### Por qué batching

Sin batching, cada sumcheck requeriría:
- Generar challenge independiente por ronda
- Verificar cada instancia separadamente

Con batching:
- Un solo challenge `α` combina todas las instancias: `∑ᵢ αⁱ · sumcheck_claimᵢ`
- Se verifica la combinación lineal en una sola pasada
- Reduce el número de operaciones de hash (challenges)

**Soundness**: Si algún sumcheck individual es inválido, la combinación lineal falla con probabilidad 1 - 1/|F| ≈ 1.

---

## 4. Generación de código Go

**Ubicación:** `gnark-transpiler/src/codegen.rs`

### 4.1 Objetivo

Convertir el grafo de nodos `MleAst` (almacenado en `NODE_ARENA`) a código Go que define un circuito Gnark. La generación debe:

1. Mapear operaciones MleAst → llamadas a `frontend.API` de Gnark
2. Aplicar CSE (Common Subexpression Elimination) para evitar recomputaciones
3. Generar struct con todas las variables de entrada
4. Generar assertions para cada constraint capturado

### 4.2 Mapeo de operaciones

| MleAst Node | Código Go generado |
|-------------|-------------------|
| `Atom::Var(idx)` | `circuit.VarName` (del struct de inputs) |
| `Atom::Scalar([u64; 4])` | Literal numérico o `bigInt("...")` si > i64::MAX |
| `Add(a, b)` | `api.Add(a, b)` |
| `Sub(a, b)` | `api.Sub(a, b)` |
| `Mul(a, b)` | `api.Mul(a, b)` |
| `Div(a, b)` | `api.Div(a, b)` |
| `Neg(e)` | `api.Neg(e)` |
| `Inv(e)` | `api.Inverse(e)` |
| `Poseidon(s, r, d)` | `poseidon.Hash(api, s, r, d)` |
| `Keccak256(e)` | `keccak.Keccak256(api, e)` |
| `ByteReverse(e)` | `poseidon.ByteReverse(api, e)` |
| `Truncate128Reverse(e)` | `poseidon.Truncate128Reverse(api, e)` |
| `Truncate128(e)` | `poseidon.Truncate128(api, e)` |
| `MulTwoPow192(e)` | `poseidon.AppendU64Transform(api, e)` |

### 4.3 Constantes grandes

Las constantes de campo son `[u64; 4]` en little-endian. Go solo tiene `int64`, así que valores grandes necesitan tratamiento especial:

```rust
fn format_scalar_for_gnark(limbs: [u64; 4]) -> String {
    // Si cabe en i64, usar literal directo
    if limbs[1] == 0 && limbs[2] == 0 && limbs[3] == 0 {
        if limbs[0] <= i64::MAX as u64 {
            return format!("{}", limbs[0]);
        }
    }

    // Convertir a decimal y usar helper bigInt
    let value = BigUint::from(limbs[3]) << 192
              + BigUint::from(limbs[2]) << 128
              + BigUint::from(limbs[1]) << 64
              + BigUint::from(limbs[0]);

    format!("bigInt(\"{}\")", value)
}
```

El código Go generado incluye el helper:

```go
func bigInt(s string) *big.Int {
    n, _ := new(big.Int).SetString(s, 10)
    return n
}
```

### 4.4 MemoizedCodeGen: CSE por reference counting

El generador usa conteo de referencias para detectar subexpresiones repetidas:

```rust
pub struct MemoizedCodeGen {
    /// Conteo de referencias para cada NodeId
    ref_counts: HashMap<usize, usize>,

    /// NodeId → nombre de variable CSE (e.g., "cse_0")
    generated: HashMap<usize, String>,

    /// Definiciones CSE en orden
    bindings: Vec<String>,

    /// Siguiente índice CSE
    cse_counter: usize,

    /// Variables de entrada encontradas
    vars: BTreeSet<u16>,

    /// Mapeo índice → nombre de variable
    var_names: HashMap<u16, String>,
}
```

#### Algoritmo en dos fases

**Fase 1: Contar referencias**

Recorre el grafo recursivamente. Cuando un nodo se visita por segunda vez, no desciende a sus hijos (ya fueron contados).

```rust
pub fn count_refs(&mut self, node_id: usize) {
    *self.ref_counts.entry(node_id).or_insert(0) += 1;

    // Solo descender a hijos en la primera visita
    if self.ref_counts[&node_id] == 1 {
        let node = get_node(node_id);
        match node {
            Node::Add(e1, e2) | Node::Mul(e1, e2) | ... => {
                self.count_refs_edge(e1);
                self.count_refs_edge(e2);
            }
            Node::Neg(e) | Node::Inv(e) | ... => {
                self.count_refs_edge(e);
            }
            Node::Poseidon(e1, e2, e3) => {
                self.count_refs_edge(e1);
                self.count_refs_edge(e2);
                self.count_refs_edge(e3);
            }
            Node::Atom(_) => {}
        }
    }
}
```

**Fase 2: Generar código**

Recorre el grafo generando expresiones Go. Si un nodo tiene `ref_count > 1`, se guarda en una variable CSE.

```rust
pub fn generate_expr(&mut self, node_id: usize) -> String {
    // Si ya generamos este nodo, retornar su nombre CSE
    if let Some(var_name) = self.generated.get(&node_id) {
        return var_name.clone();
    }

    let node = get_node(node_id);

    // Átomos se generan inline (no se hoistean)
    if let Node::Atom(atom) = node {
        return self.atom_to_gnark(atom);
    }

    // Generar expresión para este nodo
    let expr = match node {
        Node::Add(left, right) => {
            let l = self.edge_to_gnark(left);
            let r = self.edge_to_gnark(right);
            format!("api.Add({}, {})", l, r)
        }
        Node::Mul(left, right) => {
            let l = self.edge_to_gnark(left);
            let r = self.edge_to_gnark(right);
            format!("api.Mul({}, {})", l, r)
        }
        // ... otros casos
    };

    // Si tiene múltiples referencias, hoistear a variable CSE
    let ref_count = self.ref_counts.get(&node_id).copied().unwrap_or(1);
    if ref_count > 1 {
        let var_name = format!("cse_{}", self.cse_counter);
        self.cse_counter += 1;
        self.bindings.push(format!("\t{} := {}\n", var_name, expr));
        self.generated.insert(node_id, var_name.clone());
        var_name
    } else {
        expr
    }
}
```

#### Por qué threshold = 1

Usamos `ref_count > 1` como threshold para hoistear. Alternativas consideradas:

| Threshold | Pros | Contras |
|-----------|------|---------|
| `> 0` (todo) | Máximo sharing | Demasiadas variables, código ilegible |
| `> 1` (actual) | Balance | Una expresión usada 2 veces se computa 2 veces si no se hoistea |
| `> 2` | Menos variables | Más recomputación |

Con `> 1`, cada expresión referenciada múltiples veces se computa exactamente una vez.

#### Ejemplo concreto de CSE

**Grafo de entrada:**

```
assertion_0 = Sub(challenge_r0, Mul(coeff_0, eq_eval))
assertion_1 = Sub(Add(Mul(coeff_0, eq_eval), Mul(coeff_1, eq_eval_2)), claim)
                      ^^^^^^^^^^^^^^^       ← Compartido con assertion_0
```

**Sin CSE (código generado):**

```go
a_0 := api.Sub(circuit.ChallengeR0, api.Mul(circuit.Coeff0, api.Add(circuit.R0, circuit.R1)))
a_1 := api.Sub(api.Add(api.Mul(circuit.Coeff0, api.Add(circuit.R0, circuit.R1)), ...), circuit.Claim)
//                                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//                                             Recomputado
```

**Con CSE:**

```go
// CSE bindings
cse_0 := api.Add(circuit.R0, circuit.R1)           // eq_eval
cse_1 := api.Mul(circuit.Coeff0, cse_0)            // coeff_0 * eq_eval

// Assertions
a_0 := api.Sub(circuit.ChallengeR0, cse_1)
a_1 := api.Sub(api.Add(cse_1, api.Mul(circuit.Coeff1, ...)), circuit.Claim)
//                     ^^^^^
//                     Reutilizado
```

**Reducción**: En Stage 1, CSE reduce ~40% de las operaciones.

### 4.5 Tipos de assertions

El código generado soporta tres tipos de assertions:

```rust
pub enum Assertion {
    /// La expresión debe ser igual a 0
    EqualZero,

    /// La expresión debe ser igual a un input público
    EqualPublicInput { name: String },

    /// La expresión debe ser igual a otro nodo del grafo
    EqualNode(NodeId),
}
```

Generación:

```rust
match assertion {
    Assertion::EqualZero => {
        output.push_str(&format!("\tapi.AssertIsEqual({}, 0)\n", var_name));
    }
    Assertion::EqualPublicInput { name } => {
        output.push_str(&format!(
            "\tapi.AssertIsEqual({}, circuit.{})\n",
            var_name,
            sanitize_go_name(name)
        ));
    }
    Assertion::EqualNode(other_id) => {
        let other_expr = codegen.generate_expr(other_id);
        output.push_str(&format!(
            "\tapi.AssertIsEqual({}, {})\n",
            var_name, other_expr
        ));
    }
}
```

### 4.6 Sanitización de nombres

Los nombres de variables en Rust pueden tener caracteres que Go no acepta. Se sanitizan:

```rust
pub fn sanitize_go_name(name: &str) -> String {
    // Reemplazar caracteres no-alfanuméricos por _
    let cleaned: String = name
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '_' })
        .collect();

    // PascalCase cada segmento
    let parts: Vec<&str> = cleaned.split('_').filter(|s| !s.is_empty()).collect();
    parts.iter()
        .map(|s| {
            let mut c = s.chars();
            match c.next() {
                None => String::new(),
                Some(first) => first.to_uppercase().chain(c).collect(),
            }
        })
        .collect::<Vec<_>>()
        .join("_")
}

// Ejemplos:
// "stage1_r0_c0" → "Stage1_R0_C0"
// "claim::Virtual(Product)" → "Claim_Virtual_Product"
```

### 4.7 Estructura del archivo generado

```go
package jolt_verifier

import (
    "math/big"
    "github.com/consensys/gnark/frontend"
    "jolt_verifier/poseidon"
)

// Helper para constantes grandes
func bigInt(s string) *big.Int {
    n, _ := new(big.Int).SetString(s, 10)
    return n
}

// Struct con todas las variables de entrada (públicas)
type JoltStage1Circuit struct {
    // Coeficientes del sumcheck proof
    Stage1_UniSkip_Coeff0 frontend.Variable `gnark:",public"`
    Stage1_UniSkip_Coeff1 frontend.Variable `gnark:",public"`
    Stage1_R0_C0          frontend.Variable `gnark:",public"`
    Stage1_R0_C1          frontend.Variable `gnark:",public"`
    // ... más variables del proof

    // Claims de openings
    Claim_Virtual_Product_SpartanOuter frontend.Variable `gnark:",public"`
    // ... más claims

    // Commitments (hasheados al transcript)
    Commitment_0 frontend.Variable `gnark:",public"`
    // ...
}

func (circuit *JoltStage1Circuit) Define(api frontend.API) error {
    // ═══════════════════════════════════════════════════════════
    // CSE Bindings
    // ═══════════════════════════════════════════════════════════
    cse_0 := api.Add(circuit.Stage1_R0_C0, circuit.Stage1_R0_C1)
    cse_1 := poseidon.Hash(api, 0, 1, circuit.Commitment_0)
    cse_2 := poseidon.Truncate128(api, cse_1)
    // ... más bindings para subexpresiones repetidas

    // ═══════════════════════════════════════════════════════════
    // Assertions
    // ═══════════════════════════════════════════════════════════

    // Assertion 0: sumcheck consistency (round 0)
    a_0 := api.Sub(cse_0, circuit.Stage1_UniSkip_Coeff0)
    api.AssertIsEqual(a_0, 0)

    // Assertion 1: sumcheck consistency (round 1)
    a_1 := api.Sub(api.Add(cse_3, cse_4), cse_2)
    api.AssertIsEqual(a_1, 0)

    // ... más assertions

    // Assertion 31: final claim check
    a_31 := api.Sub(cse_47, circuit.Claim_Virtual_Product_SpartanOuter)
    api.AssertIsEqual(a_31, 0)

    return nil
}
```

### 4.8 Métricas Stage 1

El circuito generado para Stage 1 tiene:

| Métrica | Valor |
|---------|-------|
| Variables de entrada | ~200 |
| Variables CSE | ~150 |
| Assertions | 32 |
| Constraints R1CS | 157,607 |
| Tiempo de compilación | ~0.5s |
| Tiempo de prove (Groth16) | ~0.7s |
| Tiempo de verify | ~1.9ms |

---

## 5. Trabajo completado

### Infraestructura implementada

| Componente | Ubicación | Líneas | Función |
|------------|-----------|--------|---------|
| `MleAst` | `zklean-extractor/src/mle_ast.rs` | ~1400 | Tipo simbólico, arena global, operaciones aritméticas |
| `PoseidonAstTranscript` | `zklean-extractor/src/poseidon_ast_transcript.rs` | ~200 | Transcript Fiat-Shamir simbólico |
| `MleOpeningAccumulator` | `gnark-transpiler/src/mle_opening_accumulator.rs` | ~350 | Implementación de `OpeningAccumulator<MleAst>` |
| `AstCommitmentScheme` | `gnark-transpiler/src/ast_commitment_scheme.rs` | ~280 | Dummy commitment scheme |
| `MemoizedCodeGen` | `gnark-transpiler/src/codegen.rs` | ~400 | Generador de código Go con CSE |
| `symbolize_proof` | `gnark-transpiler/src/symbolic_proof.rs` | ~300 | Conversión proof concreto → simbólico |
| `transpile_stages` | `gnark-transpiler/src/bin/transpile_stages.rs` | ~350 | Orquestador principal |
| `TranspilableVerifier` | `jolt-core/src/zkvm/transpilable_verifier.rs` | ~560 | Verificador genérico sobre `A: OpeningAccumulator<F>` |

### Modificaciones a jolt-core

#### Trait `OpeningAccumulator<F>`

Archivo: `jolt-core/src/poly/opening_proof.rs`

Se extendió el trait para incluir métodos de escritura (antes solo tenía lectura):

```rust
pub trait OpeningAccumulator<F: JoltField> {
    // Lectura (ya existían)
    fn get_virtual_polynomial_opening(...) -> (OpeningPoint, F);
    fn get_committed_polynomial_opening(...) -> (OpeningPoint, F);
    fn get_untrusted_advice_opening(...) -> Option<(OpeningPoint, F)>;
    fn get_trusted_advice_opening(...) -> Option<(OpeningPoint, F)>;

    // Escritura (AGREGADOS para MleOpeningAccumulator)
    fn append_virtual<T: Transcript>(&mut self, ...);
    fn append_committed<T: Transcript>(&mut self, ...);
    fn append_untrusted_advice<T: Transcript>(&mut self, ...);
    fn append_trusted_advice<T: Transcript>(&mut self, ...);
    fn append_dense<T: Transcript>(&mut self, ...);
    fn append_sparse<T: Transcript>(&mut self, ...);
}
```

#### Trait `SumcheckInstanceVerifier<F, T, A>`

Archivo: `jolt-core/src/subprotocols/sumcheck_verifier.rs`

Se agregó parámetro genérico `A: OpeningAccumulator<F>`:

```rust
// ANTES:
pub trait SumcheckInstanceVerifier<F: JoltField, T: Transcript> {
    fn cache_openings(&self, acc: &mut VerifierOpeningAccumulator<F>, ...);
}

// DESPUÉS:
pub trait SumcheckInstanceVerifier<F: JoltField, T: Transcript, A: OpeningAccumulator<F>> {
    fn cache_openings(&self, acc: &mut A, ...);
}
```

Esto requirió actualizar **~25 archivos** que implementan el trait:

- `jolt-core/src/zkvm/spartan/` (6 archivos): outer, product, shift, instruction_input, claim_reductions, mod
- `jolt-core/src/zkvm/ram/` (8 archivos): val_final, val_evaluation, output_check, read_write_checking, ra_reduction, ra_virtual, hamming_booleanity, raf_evaluation
- `jolt-core/src/zkvm/registers/` (2 archivos): val_evaluation, read_write_checking
- `jolt-core/src/zkvm/instruction_lookups/` (3 archivos): read_raf_checking, ra_virtual, mod
- `jolt-core/src/zkvm/bytecode/` (2 archivos): read_raf_checking, mod
- `jolt-core/src/subprotocols/` (3 archivos): opening_reduction, booleanity, hamming_weight

### Stage 1: Funcionando

Métricas del circuito generado:

```
Variables de entrada:     ~150
Constraints (assertions): 32
R1CS constraints:         157,607
Groth16 setup time:       ~10s
Groth16 prove time:       745ms
Groth16 verify time:      1.9ms
```

Las 32 assertions corresponden a:
- 16 rondas de sumcheck × 2 assertions por ronda (consistency check + degree bound)

---

## 6. Problema actual: Stage 2

### Contexto: EqPolynomial::mle vs EqPolynomial::evals

El `EqPolynomial` es central en la verificación de sumcheck. Representa el polinomio:

```
eq(x, r) = ∏ᵢ (xᵢ · rᵢ + (1 - xᵢ)(1 - rᵢ))
```

Donde `r` es un punto fijo (típicamente challenges del transcript) y `x` son las variables.

**Dos formas de evaluar**:

#### `EqPolynomial::mle(r, point)` - Evaluación puntual

Evalúa `eq(point, r)` directamente, retornando **un solo valor**:

```rust
// En outer.rs (Stage 1):
let tau_bound_r_tail = EqPolynomial::mle(tau_low, &r_tail_reversed);
// Retorna: MleAst (una expresión simbólica)
```

Esto genera un AST pequeño: `O(n)` multiplicaciones donde `n = len(r)`.

**Stage 1 usa esto** → funciona perfectamente.

#### `EqPolynomial::evals(r)` - Expansión completa

Expande `eq(·, r)` a **todas las 2ⁿ evaluaciones** sobre el hipercubo booleano:

```rust
// En read_write_checking.rs (Stage 2):
let eq_evals: Vec<F> = EqPolynomial::evals_serial(&one_hot_challenges);
// Retorna: Vec<MleAst> de tamaño 2^n
// Para n=6: 64 expresiones simbólicas
```

Cada elemento del vector es `eq(bᵢ, r)` donde `bᵢ` es el i-ésimo punto del hipercubo {0,1}ⁿ.

**Por qué Stage 2 lo usa**:

El verificador de RAM necesita hacer una suma ponderada:
```
∑ᵢ coeffᵢ · eq(bᵢ, r)
```

Donde los `coeffᵢ` vienen del proof. En verificación normal (con `Fr`), esto es eficiente. Pero en transpilación simbólica, cada `coeffᵢ` es un valor **concreto** del proof (no simbolizado), y si `coeffᵢ = 0`, tenemos:

```
0 · eq(bᵢ, r) = Mul(Scalar(0), <symbolic expression>)
```

Sin optimización, esto crea un nodo `Mul` que Gnark luego evalúa a constante 0.

### Síntoma

Al habilitar Stage 2, Gnark rechaza el circuito:

```
Error: compiling circuit:
  constraint #53 is not satisfiable:
  non-equal constant values
```

### Análisis del error

Gnark detecta constraints donde ambos lados son constantes. Esto ocurre cuando:
1. Una expresión MleAst resulta ser enteramente constante (sin variables)
2. Se hace `AssertIsEqual(const1, const2)`

### Origen del problema

#### Código problemático

En `jolt-core/src/zkvm/ram/read_write_checking.rs`, método `expected_output_claim`:

```rust
impl<F: JoltField, T: Transcript, A: OpeningAccumulator<F>> SumcheckInstanceVerifier<F, T, A>
    for RamReadWriteCheckingVerifier<F>
{
    fn expected_output_claim(&self, accumulator: &A, sumcheck_challenges: &[F::Challenge]) -> F {
        // ...

        // Expandir EqPolynomial a 2^log_k_chunk evaluaciones
        // log_k_chunk = 6, por lo tanto 2^6 = 64 evaluaciones
        let eq_evals: Vec<F> = EqPolynomial::<F>::evals_serial(&one_hot_challenges);

        // Multiplicar cada evaluación por un coeficiente
        // PROBLEMA: algunos coeficientes son 0 (valores concretos del proof)
        let mut weighted_sum = F::zero();
        for (i, eq_eval) in eq_evals.iter().enumerate() {
            let coeff = self.coefficients[i];  // ← Puede ser 0 concreto
            weighted_sum += coeff * eq_eval;    // ← 0 * symbolic = Mul(0, symbolic)
        }

        // ...
    }
}
```

#### Flujo del problema

1. `self.coefficients[i]` es un valor **concreto** de F (viene del proof real, no simbolizado)

2. `eq_eval` es una expresión **simbólica** (MleAst derivado de challenges)

3. La multiplicación `coeff * eq_eval` cuando `coeff = 0`:
   ```rust
   // Sin optimización:
   0 * eq_eval = Mul(Scalar(0), NodeRef(eq_eval_node))
   // El resultado es un nodo Mul, no se simplifica
   ```

4. La suma de 64 términos donde todos tienen coeficiente 0:
   ```
   weighted_sum = Mul(0, e₀) + Mul(0, e₁) + ... + Mul(0, e₆₃)
   ```

5. Cuando Gnark evalúa esto:
   - Cada `Mul(0, x)` evalúa a 0
   - La suma = 0 + 0 + ... + 0 = 0
   - Es una **constante**

6. El constraint final:
   ```
   AssertIsEqual(weighted_sum + other_terms, expected_value)
   ```
   Si `weighted_sum` es constante 0 y `other_terms` también colapsa a constante, tenemos constante = constante.

#### Datos del análisis

Output del debug en `transpile_stages.rs`:

```
=== Analyzing Assertions for Constant Issues ===
  [PROBLEMATIC] Assertion 53: entirely constant!
    Structure: Sub(Add(Mul(Scalar(0), NodeRef(...)), Mul(Scalar(0), NodeRef(...)), ..., Scalar(1)), Scalar(1))
  [PROBLEMATIC] Assertion 54: entirely constant!
  ...
  Total problematic assertions: 16

=== Checking for Mul-by-Zero Pattern ===
  Found 16 assertions with mul-by-zero:
    Assertion 53: 64 mul-by-zero operations
    Assertion 54: 64 mul-by-zero operations
    ...
```

Observaciones:
- Exactamente 64 multiplicaciones por cero por assertion
- 64 = 2^6 = 2^log_k_chunk
- 16 assertions problemáticas corresponden a las 16 rondas del sumcheck de RamReadWriteChecking

### Alternativas consideradas

Antes de implementar la optimización en MleAst, se consideraron otras soluciones:

| Alternativa | Descripción | Por qué no |
|-------------|-------------|------------|
| **Simbolizar coeficientes** | Hacer que `coefficients[i]` sean MleAst en lugar de Fr concretos | Aumentaría enormemente el tamaño del circuito (64 variables extra por sumcheck) |
| **Lazy evaluation** | No crear nodos Mul hasta que se necesiten | Complejidad de implementación alta, beneficio marginal |
| **Post-processing del AST** | Recorrer el grafo y eliminar `Mul(0, x)` después | Funciona, pero es mejor prevenir que curar |
| **Simplificación en Gnark** | Dejar que Gnark optimice | Gnark detecta el problema pero no lo resuelve (genera error) |
| **Optimización en MleAst** | Interceptar `x * 0` en tiempo de construcción | ✅ Simple, efectivo, sin overhead |

**Por qué la optimización en MleAst es la mejor:**

1. **Prevención temprana**: El nodo problemático nunca se crea
2. **Zero overhead**: `is_zero()` es O(1) (un pattern match)
3. **Composable**: Se aplica automáticamente en todo el código que use MleAst
4. **Transparente**: El resto del código no necesita cambios

### Solución implementada

#### Optimización en MleAst

Archivo: `zklean-extractor/src/mle_ast.rs`

```rust
impl std::ops::Mul<&Self> for MleAst {
    type Output = Self;

    fn mul(mut self, rhs: &Self) -> Self::Output {
        // OPTIMIZACIÓN: x * 0 = 0, 0 * x = 0
        // Previene que multiplicaciones por cero creen nodos Mul
        // que luego colapsan a constantes en Gnark
        if self.is_zero() || rhs.is_zero() {
            return Self::zero();
        }
        self.binop(Node::Mul, rhs);
        self
    }
}

impl std::ops::MulAssign for MleAst {
    fn mul_assign(&mut self, rhs: Self) {
        if self.is_zero() || rhs.is_zero() {
            *self = Self::zero();
            return;
        }
        self.binop(Node::Mul, &rhs);
    }
}

impl<'a> std::ops::MulAssign<&'a Self> for MleAst {
    fn mul_assign(&mut self, rhs: &'a Self) {
        if self.is_zero() || rhs.is_zero() {
            *self = Self::zero();
            return;
        }
        self.binop(Node::Mul, rhs);
    }
}

// Helper para detectar cero
impl MleAst {
    pub fn is_zero(&self) -> bool {
        matches!(
            get_node(self.root),
            Node::Atom(Atom::Scalar(value)) if value == [0, 0, 0, 0]
        )
    }
}
```

#### Estado

- ✅ Código implementado
- ❌ **NO probado** - Stage 2 sigue comentado en `transpilable_verifier.rs`

### Posibles problemas adicionales

Si la optimización de `x * 0` no es suficiente, pueden existir otros patrones que colapsan a constantes:

```rust
// Otros patrones problemáticos potenciales:
x + 0      // Debería simplificar a x
x - 0      // Debería simplificar a x
0 - x      // Debería simplificar a -x
x * 1      // Debería simplificar a x
x / 1      // Debería simplificar a x
```

---

## 7. Trabajo pendiente

### Inmediato

1. **Probar optimización de mul-by-zero**
   - Descomentar `self.verify_stage2()?;` en `transpilable_verifier.rs`
   - Ejecutar transpilación
   - Verificar que no hay assertions constantes

2. **Si hay más patrones problemáticos**: Agregar optimizaciones para Add, Sub con 0 y Mul, Div con 1

3. **Compilar y testear circuito Go**

### Stages 3-6

Repetir el proceso para cada stage:
1. Habilitar stage
2. Transpilar
3. Analizar errores si los hay
4. Corregir
5. Testear

### Stages 7-8

Los stages 7 y 8 realizan:
- **Stage 7**: Batch opening reduction sumcheck
- **Stage 8**: Verificación de commitment (pairing check con HyperKZG/Dory)

Estos stages **no son transpilables automáticamente** porque:

1. Usan métodos específicos de `VerifierOpeningAccumulator` que no están en el trait:
   - `prepare_for_sumcheck()`
   - `verify_batch_opening_sumcheck()`
   - `finalize_batch_opening_sumcheck()`
   - `verify_stage8()` (pairing check)

2. Stage 8 requiere operaciones de curva elíptica (pairings) que MleAst no puede representar

#### Qué hace Stage 7 (Batch Opening Reduction)

**Propósito**: Reducir múltiples claims de openings a un solo claim verificable.

Durante stages 1-6, el accumulator acumuló N claims de la forma:
```
claim_i: p_i(r_i) = v_i
```

Stage 7 los combina con random linear combination:
```
∑_i α^i · p_i(r_i) = ∑_i α^i · v_i
```

Esto se verifica con un sumcheck sobre el polinomio combinado.

**Operaciones involucradas:**
1. Obtener challenge α del transcript
2. Computar combinación lineal de claims
3. Ejecutar sumcheck de reducción
4. El resultado es un único claim sobre un punto "combinado"

#### Qué hace Stage 8 (Pairing Check)

**Propósito**: Verificar que los commitments corresponden a los polinomios evaluados.

Para HyperKZG, esto involucra:
```
e(C - [v]G₁, G₂) = e(π, [τ - r]G₂)
```

Donde:
- `C` = commitment al polinomio (punto en G₁)
- `v` = valor del claim
- `r` = punto de evaluación
- `π` = proof de opening (punto en G₁)
- `e(·,·)` = pairing bilineal

**Por qué no es transpilable:**
- Los puntos de curva (C, π, G₁, G₂) no son elementos del campo escalar
- Las operaciones de pairing no se pueden expresar como operaciones de campo
- MleAst solo representa elementos de Fr (campo escalar)

#### Implementación manual en Go

**Stage 7** podría transpilarse parcialmente:
- El sumcheck interno es similar a stages 1-6
- La combinación lineal de claims es aritmética de campo

```go
// Pseudocódigo Stage 7 en Go
func VerifyStage7(api frontend.API, claims []Claim, alpha frontend.Variable) {
    // 1. Combinar claims con RLC
    combined := frontend.Variable(0)
    alphaPow := frontend.Variable(1)
    for _, claim := range claims {
        combined = api.Add(combined, api.Mul(alphaPow, claim.Value))
        alphaPow = api.Mul(alphaPow, alpha)
    }

    // 2. Verificar sumcheck (similar a stages 1-6)
    verifySumcheck(api, combinedPolynomial, combined, ...)
}
```

**Stage 8** requiere primitivas de Gnark para pairings:

```go
import "github.com/consensys/gnark/std/algebra/emulated/sw_bn254"

func VerifyStage8(api frontend.API, commitment, proof sw_bn254.G1Affine, ...) {
    // Gnark tiene soporte nativo para pairing checks
    pairing, _ := sw_bn254.NewPairing(api)

    // Verificar e(C - [v]G₁, G₂) = e(π, [τ - r]G₂)
    lhs := pairing.Pair([]*sw_bn254.G1Affine{...}, []*sw_bn254.G2Affine{...})
    rhs := pairing.Pair([]*sw_bn254.G1Affine{...}, []*sw_bn254.G2Affine{...})

    pairing.AssertIsEqual(lhs, rhs)
}
```

**Nota**: Gnark usa "emulated" arithmetic para curvas que no son nativas. Para BN254 sobre BN254, esto es eficiente. Para otras curvas (BLS12-381, etc.) hay overhead.

**Solución**: Implementar stages 7-8 manualmente en Go usando las primitivas de Gnark para:
- Verificación de batch openings
- Pairing checks (Gnark tiene soporte nativo)

### Integración final

1. Circuito completo stages 1-6 funcionando
2. Implementación manual Go para stages 7-8
3. Exportar verificador Solidity: `vk.ExportSolidity(file)`
4. Integrar con flujo prover Jolt → Gnark → on-chain

---

## 8. Procedimiento para continuar

### Paso 1: Probar optimización actual

```bash
# 1. Editar transpilable_verifier.rs línea 223
#    Cambiar:  // self.verify_stage2()?;
#    A:        self.verify_stage2()?;

# 2. Compilar
cd /path/to/jolt
cargo build -p gnark-transpiler -p zklean-extractor -p jolt-core --release

# 3. Ejecutar transpilación
cargo run -p gnark-transpiler --release --bin transpile_stages 2>&1 | tee transpile.log

# 4. Buscar en output:
#    "Total problematic assertions: X" → debe ser 0
#    "Found X assertions with mul-by-zero" → debe ser 0
grep -E "(problematic|mul-by-zero)" transpile.log
```

### Paso 2: Si hay errores, agregar más optimizaciones

Editar `zklean-extractor/src/mle_ast.rs`:

```rust
// Agregar is_one()
impl MleAst {
    pub fn is_one(&self) -> bool {
        matches!(
            get_node(self.root),
            Node::Atom(Atom::Scalar(value)) if value == [1, 0, 0, 0]
        )
    }
}

// Optimizar Add
impl std::ops::Add<&Self> for MleAst {
    fn add(mut self, rhs: &Self) -> Self::Output {
        if self.is_zero() { return rhs.clone(); }
        if rhs.is_zero() { return self; }
        self.binop(Node::Add, rhs);
        self
    }
}

// Optimizar Sub
impl std::ops::Sub<&Self> for MleAst {
    fn sub(mut self, rhs: &Self) -> Self::Output {
        if rhs.is_zero() { return self; }
        // Nota: 0 - x = -x, pero eso requiere crear Neg node
        self.binop(Node::Sub, rhs);
        self
    }
}

// Optimizar Mul con 1
impl std::ops::Mul<&Self> for MleAst {
    fn mul(mut self, rhs: &Self) -> Self::Output {
        if self.is_zero() || rhs.is_zero() { return Self::zero(); }
        if self.is_one() { return rhs.clone(); }
        if rhs.is_one() { return self; }
        self.binop(Node::Mul, rhs);
        self
    }
}
```

### Paso 3: Testear circuito Go

```bash
cd gnark-transpiler/go

# Compilar
go build ./...

# Ejecutar test (puede tardar varios minutos)
go test -v -run TestStages16Circuit -timeout 600s
```

### Paso 4: Interpretar errores de Go

| Error | Causa | Solución |
|-------|-------|----------|
| `non-equal constant values` | Hay assertions constantes | Agregar más optimizaciones en MleAst |
| `Poseidon hash mismatch` | Parámetros Poseidon incorrectos | Revisar `go/poseidon/poseidon.go` |
| `constraint not satisfied` | Witness incorrecto | Regenerar bundle JSON |
| `undefined: X` | Falta variable en struct | Revisar generación de struct |

### Paso 5: Continuar con más stages

```bash
# Para cada stage N en [3, 4, 5, 6]:

# 1. Editar transpilable_verifier.rs
#    Descomentar: self.verify_stageN()?;

# 2. Recompilar y transpilar
cargo build -p gnark-transpiler --release
cargo run -p gnark-transpiler --release --bin transpile_stages

# 3. Analizar output
# 4. Corregir si hay errores
# 5. Testear en Go
```

---

## Apéndice A: Estructura de archivos

### Rust

```
jolt/
├── jolt-core/src/
│   ├── poly/
│   │   └── opening_proof.rs          # Trait OpeningAccumulator
│   ├── subprotocols/
│   │   ├── sumcheck.rs               # BatchedSumcheck::verify
│   │   └── sumcheck_verifier.rs      # Trait SumcheckInstanceVerifier
│   └── zkvm/
│       ├── transpilable_verifier.rs  # Verificador genérico
│       ├── spartan/                  # Sumcheck verifiers stage 1-3
│       ├── ram/                      # Sumcheck verifiers RAM
│       ├── registers/                # Sumcheck verifiers registros
│       ├── instruction_lookups/      # Sumcheck verifiers lookups
│       └── bytecode/                 # Sumcheck verifiers bytecode
│
├── zklean-extractor/src/
│   ├── mle_ast.rs                    # Tipo simbólico MleAst
│   ├── poseidon_ast_transcript.rs    # Transcript simbólico
│   └── lib.rs
│
└── gnark-transpiler/src/
    ├── bin/
    │   └── transpile_stages.rs       # Programa principal
    ├── mle_opening_accumulator.rs    # MleOpeningAccumulator
    ├── ast_commitment_scheme.rs      # AstCommitmentScheme
    ├── codegen.rs                    # MemoizedCodeGen
    ├── symbolic_proof.rs             # symbolize_proof()
    └── lib.rs
```

### Go

```
gnark-transpiler/go/
├── stages16_circuit.go               # Circuito generado (stages 1-6)
├── stage1_circuit.go                 # Circuito stage 1 solo
├── helpers.go                        # bigInt() helper
├── stage1_bundle.json                # Witness values para test
├── stages16_circuit_test.go          # Test del circuito
├── stage1_circuit_test.go            # Test stage 1
└── poseidon/
    └── poseidon.go                   # Implementación Poseidon
```

---

## Apéndice B: Comandos útiles

```bash
# Ver número de constraints en circuito generado
grep -c "api.AssertIsEqual" gnark-transpiler/go/stages16_circuit.go

# Ver número de variables
grep -c "frontend.Variable" gnark-transpiler/go/stages16_circuit.go

# Ver tamaño del circuito
wc -l gnark-transpiler/go/stages16_circuit.go

# Regenerar proof de prueba
cd jolt/examples/fib && cargo run --release

# Limpiar y recompilar
cargo clean -p gnark-transpiler -p zklean-extractor
cargo build -p gnark-transpiler -p zklean-extractor --release

# Ejecutar con backtrace
RUST_BACKTRACE=1 cargo run -p gnark-transpiler --release --bin transpile_stages

# Profiling de memoria del circuito Go
go test -v -run TestStages16Circuit -memprofile mem.out
go tool pprof mem.out
```

