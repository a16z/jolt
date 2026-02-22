import zkLean

def mux [ZKField f]
  (cases : Array (ZKExpr f × ZKExpr f))
  : ZKExpr f :=
  Array.foldl (fun acc (flag, term) => acc + flag * term) 0 cases

def mux_mles [ZKField f]
  (interleaving : Interleaving)
  (left right : ZKExpr f)
  (mle_cases : Array (ZKExpr f × ((Vector f 128) -> f)))
  : ZKBuilder f (ZKExpr f) := do
  let cases <- mle_cases.mapM (fun (flag, table) => do
    let mle : LookupTableMLE f (64 + 64) := .mk interleaving table
    let term <- ZKBuilder.lookup_mle mle left right
    pure (flag, term))
  pure (mux cases)
