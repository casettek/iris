use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;

use crate::*;

#[test]
fn an_output_is_the_first_form_it_matches() {
    let forms = vec![
        Form {
            text: r#"{"verdict":"confirmed"}"#.to_string(),
            code: 1,
        },
        Form {
            text: r#"{"verdict":"denied"}"#.to_string(),
            code: 2,
        },
        Form {
            text: r#"{"verdict":"denied"}"#.to_string(),
            code: 3,
        },
    ];
    let code = |forms: &[Form], output: &str| run_forms(forms, output).code;
    assert_eq!(code(&forms, r#"{"verdict":"confirmed"}"#), 1);
    assert_eq!(code(&forms, "  {\"verdict\":\"denied\"}\n"), 2);
    assert_eq!(code(&forms, r#"{"verdict": "denied"}"#), 0);
    assert_eq!(code(&forms, ""), 0);
    assert_eq!(code(&[], "anything"), 0);
    let long = vec![Form {
        text: "x".repeat(MAX_FORM + 1),
        code: 5,
    }];
    assert_eq!(code(&long, &"x".repeat(MAX_FORM + 1)), 0);

    let id = |forms: &[Form], output: &str| forms_id(&run_forms(forms, output));
    let stated = id(&forms, "");
    assert_eq!(id(&forms, "another output"), stated);
    assert_ne!(id(&forms[..2], ""), stated);
    assert_ne!(id(&[], ""), stated);
    let mut other = forms.clone();
    other[0].code = 9;
    assert_ne!(id(&other, ""), stated);

    // The record carries the claim record it was given, untouched.
    let record = close_result(
        run_forms(&forms, r#"{"verdict":"denied"}"#),
        &[1; 32],
        &[2; 32],
        true,
    );
    assert_eq!(record.statement.as_slice(), [1; 32]);
    assert_eq!(record.output_text_sha256.as_slice(), [2; 32]);
    assert!(record.valid);
    assert_eq!(record.code, 2);
    assert_eq!(record.forms_sha256, stated);
}

#[test]
fn two_outputs_agree_when_they_are_the_same_form() {
    let forms: Vec<Form> = [("yes", 1), ("no", 2)]
        .into_iter()
        .map(|(text, code)| Form {
            text: text.to_string(),
            code,
        })
        .collect();
    let claimed = |valid| Claimed {
        statement: &[7; 32],
        output_text_sha256: &[9; 32],
        valid,
    };
    let run = |forms: &[Form], first: &str, second: &str, same_input, valid: [bool; 2]| {
        close_consensus(
            run_consensus(forms, first, second, same_input),
            claimed(valid[0]),
            claimed(valid[1]),
        )
    };
    let honest = [true, true];

    // The same form, whatever whitespace is around each output.
    let record = run(&forms, "yes", " yes\n", true, honest);
    assert!(record.agree && record.same_input);
    assert_eq!(
        (record.code, record.first.code, record.second.code),
        (1, 1, 1)
    );
    assert_eq!(record.first.forms_sha256, record.second.forms_sha256);
    assert_eq!(record.first.forms_sha256, forms_id(&run_forms(&forms, "")));
    assert_eq!(consensus_description(&record), "agree");

    // Two forms, and one output that is no form.
    let record = run(&forms, "yes", "no", true, honest);
    assert!(!record.agree);
    assert_eq!(
        (record.code, record.first.code, record.second.code),
        (0, 1, 2)
    );
    assert_eq!(consensus_description(&record), "disagree");
    let record = run(&forms, "yes", "yes, probably", true, honest);
    assert_eq!((record.agree, record.second.code), (false, 0));
    // Two outputs that are the same text and no form have no code to share.
    let record = run(&forms, "maybe", "maybe", true, honest);
    assert_eq!((record.agree, record.code), (false, 0));

    // With no forms the texts themselves are compared.
    let record = run(&[], "The answer is 4.", "The answer is 4.\n", true, honest);
    assert_eq!((record.agree, record.code), (true, 0));
    assert!(!run(&[], "4", "four", true, honest).agree);

    // Agreement is of two valid records over one input.
    let record = run(&forms, "yes", "yes", false, honest);
    assert_eq!(
        (record.agree, record.same_input, record.code),
        (false, false, 0)
    );
    assert_eq!(
        consensus_description(&record),
        "no consensus: the inferences name different inputs"
    );
    for valid in [[false, true], [true, false]] {
        let record = run(&forms, "yes", "yes", true, valid);
        assert_eq!((record.agree, record.code), (false, 0));
        assert_eq!(
            (record.first.valid, record.second.valid),
            (valid[0], valid[1])
        );
        assert_eq!(
            consensus_description(&record),
            "no consensus: a claim record is not valid"
        );
    }

    assert!(same_input("a", "s", "a", "s"));
    assert!(!same_input("a", "s", "b", "s"));
    assert!(!same_input("a", "s", "a", "t"));
}
