use syn::{
    braced,
    parse::{Parse, ParseStream},
    Expr, Ident, Path, Token,
};

/// Top-level attribute: `#[sumcheck_claims(input_claim = ..., output_claim = ...)]`
#[derive(Debug)]
pub struct SumcheckClaimsAttr {
    pub input_claim: ClaimSpec,
    pub output_claim: ClaimSpec,
}

impl Parse for SumcheckClaimsAttr {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut input_claim = None;
        let mut output_claim = None;

        while !input.is_empty() {
            let ident: Ident = input.parse()?;
            input.parse::<Token![=]>()?;

            match ident.to_string().as_str() {
                "input_claim" => {
                    input_claim = Some(parse_claim_spec(input)?);
                }
                "output_claim" => {
                    output_claim = Some(parse_claim_spec(input)?);
                }
                other => {
                    return Err(syn::Error::new(
                        ident.span(),
                        format!("unknown attribute: {other}"),
                    ));
                }
            }

            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        }

        Ok(SumcheckClaimsAttr {
            input_claim: input_claim
                .ok_or_else(|| syn::Error::new(input.span(), "missing input_claim"))?,
            output_claim: output_claim
                .ok_or_else(|| syn::Error::new(input.span(), "missing output_claim"))?,
        })
    }
}

/// Parse either `None` or `{ ... }` block for claim spec
fn parse_claim_spec(input: ParseStream) -> syn::Result<ClaimSpec> {
    if input.peek(Ident) {
        let ident: Ident = input.fork().parse()?;
        if ident == "None" {
            input.parse::<Ident>()?;
            return Ok(ClaimSpec::None);
        }
    }

    let content;
    braced!(content in input);
    let spec = ClaimSpecContent::parse(&content)?;
    Ok(ClaimSpec::Some(Box::new(spec)))
}

#[derive(Debug)]
pub enum ClaimSpec {
    None,
    Some(Box<ClaimSpecContent>),
}

/// Content of a claim spec block
#[derive(Debug)]
pub struct ClaimSpecContent {
    pub points: Vec<(Ident, PolynomialRef)>,
    pub openings: Vec<(Ident, PolynomialRef)>,
    pub expr: Expr,
    pub derived: Vec<(Ident, Expr)>,
}

impl Parse for ClaimSpecContent {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut points = Vec::new();
        let mut openings = Vec::new();
        let mut expr = None;
        let mut derived = Vec::new();

        while !input.is_empty() {
            let ident: Ident = input.parse()?;
            input.parse::<Token![:]>()?;

            match ident.to_string().as_str() {
                "points" => {
                    points = parse_openings_map(input)?;
                }
                "openings" => {
                    openings = parse_openings_map(input)?;
                }
                "expr" => {
                    expr = Some(input.parse::<Expr>()?);
                }
                "derived" => {
                    // Parse: name = expr (, name = expr)*
                    derived = parse_derived(input)?;
                }
                other => {
                    return Err(syn::Error::new(
                        ident.span(),
                        format!("unknown field: {other}"),
                    ));
                }
            }

            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        }

        Ok(ClaimSpecContent {
            points,
            openings,
            expr: expr.ok_or_else(|| syn::Error::new(input.span(), "missing expr"))?,
            derived,
        })
    }
}

/// Parse `{ name: (Poly, Stage), ... }`
fn parse_openings_map(input: ParseStream) -> syn::Result<Vec<(Ident, PolynomialRef)>> {
    let content;
    braced!(content in input);

    let mut result = Vec::new();
    while !content.is_empty() {
        let name: Ident = content.parse()?;
        content.parse::<Token![:]>()?;

        let tuple_content;
        syn::parenthesized!(tuple_content in content);
        let poly: Path = tuple_content.parse()?;
        tuple_content.parse::<Token![,]>()?;
        let stage: Path = tuple_content.parse()?;

        result.push((name, PolynomialRef { poly, stage }));

        if content.peek(Token![,]) {
            content.parse::<Token![,]>()?;
        }
    }

    Ok(result)
}

/// Parse derived values: `name = expr (, name = expr)*`
fn parse_derived(input: ParseStream) -> syn::Result<Vec<(Ident, Expr)>> {
    let mut result = Vec::new();

    loop {
        let name: Ident = input.parse()?;
        input.parse::<Token![=]>()?;
        let expr: Expr = input.parse()?;
        result.push((name, expr));

        // Check for comma separating multiple derived values
        if input.peek(Token![,]) {
            let lookahead = input.fork();
            lookahead.parse::<Token![,]>()?;
            // If next token is an identifier followed by `=`, it's another derived
            if lookahead.peek(Ident) {
                let _: Ident = lookahead.fork().parse().unwrap();
                let after_ident = lookahead.fork();
                after_ident.parse::<Ident>().ok();
                if after_ident.peek(Token![=]) {
                    input.parse::<Token![,]>()?;
                    continue;
                }
            }
        }
        break;
    }

    Ok(result)
}

#[derive(Debug, Clone)]
pub struct PolynomialRef {
    pub poly: Path,
    pub stage: Path,
}
