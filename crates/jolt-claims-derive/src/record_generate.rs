use proc_macro2::TokenStream;
use quote::quote;
use syn::{parse2, parse_quote, Error, Expr, ItemFn, Pat, Result, Stmt};

fn path(expression: &Expr, name: &str) -> bool {
    matches!(expression, Expr::Path(value) if value.path.is_ident(name))
}

pub fn expand(input: TokenStream) -> Result<TokenStream> {
    let mut function: ItemFn = parse2(input)?;
    if function.sig.ident != "collect_guest_proofs" {
        return Err(Error::new_spanned(
            function.sig,
            "recording is restricted to canonical proof collection",
        ));
    }
    let Some(Stmt::Expr(last, None)) = function.block.stmts.last() else {
        return Err(Error::new_spanned(
            &function.block,
            "missing canonical stream return",
        ));
    };
    if !path(last, "all_groups_data") {
        return Err(Error::new_spanned(last, "canonical stream return changed"));
    }
    let mut found = 0;
    let mut statements = vec![parse_quote!(
        let mut __recipe_request = crate::immutable_recipe_generation::request(guest, proofs, mode)
            .expect("validated native immutable recipe request");
    )];
    let mut inputs = 0;
    let mut traces = 0;
    for mut statement in function.block.stmts {
        if let Stmt::Local(local) = &mut statement {
            if matches!(&local.pat, Pat::Ident(name) if name.ident == "inputs") {
                let Some(initializer) = &mut local.init else {
                    return Err(Error::new_spanned(local, "missing input initializer"));
                };
                let Expr::MethodCall(call) = &*initializer.expr else {
                    return Err(Error::new_spanned(
                        &initializer.expr,
                        "unexpected canonical inputs",
                    ));
                };
                if call.method != "inputs"
                    || !path(&call.receiver, "guest")
                    || call.args.len() != 1
                    || !call.args.first().is_some_and(|arg| path(arg, "proofs"))
                {
                    return Err(Error::new_spanned(
                        call,
                        "canonical input construction changed",
                    ));
                }
                let original = &initializer.expr;
                *initializer.expr = parse_quote!(
                    crate::immutable_recipe_generation::inputs(&__recipe_request, || #original)
                );
                inputs += 1;
            }
        }
        if let Stmt::Expr(Expr::ForLoop(loop_body), _) = &mut statement {
            if path(&loop_body.expr, "inputs") {
                if !matches!(&*loop_body.pat, Pat::Ident(name) if name.ident == "input_bytes") {
                    return Err(Error::new_spanned(
                        loop_body,
                        "canonical input loop changed",
                    ));
                }
                let mut body = Vec::new();
                for step in std::mem::take(&mut loop_body.body.stmts) {
                    let observe = matches!(&step, Stmt::Local(local)
                        if matches!(&local.pat, Pat::Ident(name) if name.ident == "trace"));
                    if observe {
                        let Stmt::Local(local) = &step else {
                            return Err(Error::new_spanned(&step, "missing canonical trace"));
                        };
                        let Some(initializer) = &local.init else {
                            return Err(Error::new_spanned(local, "missing trace initializer"));
                        };
                        let Expr::MethodCall(expect) = &*initializer.expr else {
                            return Err(Error::new_spanned(
                                &initializer.expr,
                                "canonical trace result changed",
                            ));
                        };
                        if expect.method != "expect"
                            || !matches!(&*expect.receiver, Expr::MethodCall(call)
                            if call.method == "trace" && call.args.len() == 2)
                        {
                            return Err(Error::new_spanned(expect, "canonical trace call changed"));
                        }
                    }
                    body.push(step);
                    if observe {
                        traces += 1;
                        body.push(parse_quote!(
                            if let Some(request) = __recipe_request.as_mut() {
                                request
                                    .observe_trace(trace.dimensions.trace_length)
                                    .expect("one canonical unpadded trace observation");
                            }
                        ));
                    }
                }
                loop_body.body.stmts = body;
            }
        }
        if let Stmt::Expr(Expr::ForLoop(loop_body), _) = &statement {
            if matches!(&*loop_body.expr, Expr::Reference(reference) if path(&reference.expr, "records"))
            {
                let Pat::Tuple(pattern) = &*loop_body.pat else {
                    return Err(Error::new_spanned(
                        loop_body,
                        "unexpected canonical proof binding",
                    ));
                };
                if pattern.elems.len() != 2 || !pattern.elems.iter().zip(["proof", "public_io"])
                    .all(|(pattern, name)| matches!(pattern, Pat::Ident(value) if value.ident == name)) {
                    return Err(Error::new_spanned(loop_body, "canonical proof binding changed"));
                }
                let Some(Stmt::Expr(Expr::MethodCall(expect), Some(_))) =
                    loop_body.body.stmts.first()
                else {
                    return Err(Error::new_spanned(
                        loop_body,
                        "missing canonical verification result check",
                    ));
                };
                let Expr::Call(verify) = &*expect.receiver else {
                    return Err(Error::new_spanned(
                        expect,
                        "unexpected verification receiver",
                    ));
                };
                let Expr::Path(verify_path) = &*verify.func else {
                    return Err(Error::new_spanned(verify, "unexpected verification callee"));
                };
                let names: Vec<_> = verify_path
                    .path
                    .segments
                    .iter()
                    .map(|segment| segment.ident.to_string())
                    .collect();
                if expect.method != "expect"
                    || names != ["jolt_sdk", "jolt_verifier", "verify"]
                    || verify.args.len() != 4
                {
                    return Err(Error::new_spanned(
                        verify,
                        "canonical verifier invocation changed",
                    ));
                }
                let mut arguments = verify.args.iter();
                if !matches!(arguments.next(), Some(Expr::Reference(reference)) if path(&reference.expr, "attached"))
                    || !arguments.next().is_some_and(|arg| path(arg, "public_io"))
                    || !arguments.next().is_some_and(|arg| path(arg, "proof"))
                    || !arguments.next().is_some_and(|arg| path(arg, "None"))
                {
                    return Err(Error::new_spanned(
                        verify,
                        "canonical verifier inputs changed",
                    ));
                }
                found += 1;
                statements.push(
                    parse_quote!(let __recipe_session = crate::immutable_recipe_generation::begin(
                    __recipe_request, n, records.len(), max_trace_length,
                );),
                );
                statements.push(statement);
                statements.push(
                    parse_quote!(let __recipe_capture = __recipe_session.map(|session| {
                    session.finish(&records, &verifier_preprocessing, &payloads)
                });),
                );
                continue;
            }
        }
        statements.push(statement);
    }
    if found != 1 || inputs != 1 || traces != 1 {
        return Err(Error::new_spanned(
            &function.sig,
            "exactly one canonical input, trace and verification loop is required",
        ));
    }
    let Some(last) = statements.last_mut() else {
        return Err(Error::new_spanned(&function.sig, "missing stream return"));
    };
    *last = parse_quote!({
        crate::immutable_recipe_generation::publish(__recipe_capture, &all_groups_data)
            .expect("canonical immutable recipe generation");
        all_groups_data
    });
    function.block.stmts = statements;
    Ok(quote!(#function))
}
