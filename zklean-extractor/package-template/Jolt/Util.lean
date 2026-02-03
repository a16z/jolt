import ZKLean

def mux [ZKField f]
  (cases : Array (ZKExpr f × ZKExpr f))
  : ZKBuilder f (ZKExpr f) :=
  Array.foldl (fun acc (flag, term) => do
    let a : ZKExpr f <- acc
    pure (a + flag * term)) (pure 0) cases

def mux_mles [ZKField f]
  (interleaving : Interleaving)
  (left right : ZKExpr f)
  (mle_cases : Array (ZKExpr f × ((Vector f 128) -> f)))
  : ZKBuilder f (ZKExpr f) := do
  let cases : Array (ZKExpr f × ZKExpr f) <- mle_cases.foldl (fun acc ((flag, table) : (ZKExpr f × (Vector f 128 -> f))) => do
    let mle : LookupTableMLE f (64 + 64) := .mk interleaving table
    let term <- ZKBuilder.lookup_mle mle left right
    let a <- acc
    pure (a ++ #[(flag, term)])) (pure #[] : ZKBuilder f (Array (ZKExpr f × ZKExpr f)))
  mux cases
