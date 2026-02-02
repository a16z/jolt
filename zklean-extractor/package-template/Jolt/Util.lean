import ZKLean

-- XXX We keep this here for now because zklean does not yet handle 64-bit MLEs in lookup_mle
def lookup_mle_64bit [ZKField f] (tbl : LookupTableMLE f 128) (left right : ZKExpr f) : ZKBuilder f (ZKExpr f) :=
  let res := evalLookupTableMLE tbl
    (ZKField.field_to_bits (num_bits := 64) left.eval)
    (ZKField.field_to_bits (num_bits := 64) right.eval)
  pure (ZKExpr.Field res)

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
    let mle := LookupTableMLE.mk interleaving table
    let term <- lookup_mle_64bit mle left right
    let a <- acc
    pure (a ++ #[(flag, term)])) (pure #[] : ZKBuilder f (Array (ZKExpr f × ZKExpr f)))
  mux cases
