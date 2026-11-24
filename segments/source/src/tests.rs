use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

use crate::*;
use raster::BytesPage;

const SECRET: [u8; 32] = [7; 32];

const DOCUMENT: &str = r#"{"type":"Feature","properties":{"mag":5.1,"place":"10 km \"N\" of {X}, Tōhoku 東北","magnitude":7,"nested":{"mag":9,"list":[{"mag":8},"}"]},"time":1763240000000,"flag":true,"none":null},"geometry":{"coordinates":[1,2,3]},"id":"us7000abcd"}"#;

/// A path written with `.` between its steps.
fn steps(path: &str) -> String {
    path.replace('.', "\n")
}

fn value_at(document: &[u8], path: &str, page: usize) -> FieldValue {
    let path = steps(path);
    let mut scan = begin_field_scan(&path);
    for page in document.chunks(page) {
        scan = scan_field_page(scan, page, &path);
    }
    finish_field_scan(scan)
}

fn found(document: &str, path: &str) -> String {
    let value = value_at(document.as_bytes(), path, DOCUMENT_PAGE);
    assert_eq!(value.fault, FAULT_NONE, "{path}");
    value.value
}

fn fault(document: &str, path: &str) -> u32 {
    let value = value_at(document.as_bytes(), path, DOCUMENT_PAGE);
    assert_eq!(value.value, "", "{path}");
    value.fault
}

#[test]
fn a_fold_in_any_pieces_is_the_hash_of_the_whole() {
    let bytes: Vec<u8> = (0..300u32).map(|i| (i * 7 + 3) as u8).collect();
    for length in [0, 1, 55, 56, 63, 64, 65, 119, 120, 128, 300] {
        let whole = sha256(&bytes[..length]);
        for piece in [1, 3, 64, 100] {
            let mut fold = begin_byte_fold();
            for part in bytes[..length].chunks(piece) {
                fold = fold_bytes(fold, part);
            }
            assert_eq!(fold_sha256(&fold), whole);
            assert_eq!(fold.total, length as u64);
        }
    }
    assert_eq!(unhex(&hex(&bytes)).unwrap(), bytes);
    assert_eq!(unhex("AB"), None);
    assert_eq!(unhex("abc"), None);
}

#[test]
fn a_value_is_the_documents_own_bytes() {
    assert_eq!(found(DOCUMENT, "properties.mag"), "5.1");
    assert_eq!(
        found(DOCUMENT, "properties.place"),
        r#""10 km \"N\" of {X}, Tōhoku 東北""#
    );
    assert_eq!(found(DOCUMENT, "properties.magnitude"), "7");
    assert_eq!(found(DOCUMENT, "properties.nested.mag"), "9");
    assert_eq!(found(DOCUMENT, "properties.time"), "1763240000000");
    assert_eq!(found(DOCUMENT, "properties.flag"), "true");
    assert_eq!(found(DOCUMENT, "properties.none"), "null");
    assert_eq!(found(DOCUMENT, "type"), r#""Feature""#);
    assert_eq!(found(DOCUMENT, "id"), r#""us7000abcd""#);
}

#[test]
fn a_path_that_is_not_there_is_missing() {
    for path in [
        "mag",
        "properties.ma",
        "properties.magn",
        "properties.nope",
        "properties.mag.x",
        "properties.nested.list.mag",
        "nested.mag",
        "id.x",
        "none",
    ] {
        assert_eq!(fault(DOCUMENT, path), FAULT_FIELD_MISSING, "{path}");
    }
    for document in [
        "",
        "   ",
        "[1]",
        "5",
        r#"{"a":"#,
        r#"{"a" 1}"#,
        r#"{"b":{"a":1}"#,
    ] {
        assert_eq!(fault(document, "a"), FAULT_FIELD_MISSING, "{document}");
    }
}

#[test]
fn an_object_or_array_is_not_a_value() {
    assert_eq!(fault(DOCUMENT, "properties"), FAULT_FIELD_NOT_SCALAR);
    assert_eq!(fault(DOCUMENT, "properties.nested"), FAULT_FIELD_NOT_SCALAR);
    assert_eq!(
        fault(DOCUMENT, "geometry.coordinates"),
        FAULT_FIELD_NOT_SCALAR
    );
    assert_eq!(fault(r#"{"a":tru}"#, "a"), FAULT_FIELD_NOT_SCALAR);
    assert_eq!(fault(r#"{"a":1x}"#, "a"), FAULT_FIELD_NOT_SCALAR);
}

#[test]
fn a_path_must_have_keys() {
    let long = "a".repeat(MAX_PATH + 1);
    for path in ["", ".", "a.", ".a", "a..b", long.as_str()] {
        assert_eq!(fault(DOCUMENT, path), FAULT_RECIPE, "{path}");
    }
}

#[test]
fn where_pages_are_cut_changes_nothing() {
    let pretty = "{\n  \"a\" : {\r\n\t\"b\" : [ \"x\" , { \"c\" : 1 } ] ,\n  \"c\" :  -1.5e+3\n  } ,\n \"d\" : \"é\"\n}\n";
    for (document, paths) in [
        (
            DOCUMENT,
            vec![
                "properties.mag",
                "properties.place",
                "properties.nested.mag",
                "properties.none",
                "id",
                "mag",
                "properties",
                "properties.nope",
            ],
        ),
        (pretty, vec!["a.c", "d", "a.b", "a.x", "c"]),
    ] {
        for path in paths {
            let whole = value_at(document.as_bytes(), path, document.len());
            for page in 1..=document.len() {
                assert_eq!(
                    value_at(document.as_bytes(), path, page),
                    whole,
                    "{path} at {page}"
                );
            }
        }
    }
    assert_eq!(found(pretty, "a.c"), "-1.5e+3");
    assert_eq!(found(pretty, "d"), "\"é\"");
}

#[test]
fn the_first_member_with_the_key_is_taken() {
    assert_eq!(found(r#"{"a":1,"a":2}"#, "a"), "1");
    assert_eq!(found(r#"{"a":{"b":1},"a":{"b":2}}"#, "a.b"), "1");
    // The first `a` has no `b`, and the scan does not go on to the second.
    assert_eq!(
        fault(r#"{"a":{"c":1},"a":{"b":2}}"#, "a.b"),
        FAULT_FIELD_MISSING
    );
}

#[test]
fn a_key_is_compared_as_written() {
    assert_eq!(found(r#"{"a\"b":1,"x":2}"#, r#"a\"b"#), "1");
    assert_eq!(found(r#"{"a\\":1,"x":2}"#, "x"), "2");
    // A key written as an escape is not the key it decodes to.
    let escaped = format!("{{\"{}u0061\":1}}", '\\');
    assert_eq!(fault(&escaped, "a"), FAULT_FIELD_MISSING);
    assert_eq!(found(r#"{"x":"\\","a":3}"#, "a"), "3");
    assert_eq!(found(r#"{"x":["\"]}"],"a":4}"#, "a"), "4");
}

#[test]
fn a_value_has_a_size_and_is_text() {
    let most = format!(r#"{{"a":"{}"}}"#, "x".repeat(MAX_VALUE - 2));
    assert_eq!(found(&most, "a").len(), MAX_VALUE);
    let over = format!(r#"{{"a":"{}"}}"#, "x".repeat(MAX_VALUE - 1));
    assert_eq!(fault(&over, "a"), FAULT_FIELD_TOO_LONG);
    let digits = format!(r#"{{"a":{}}}"#, "1".repeat(MAX_VALUE + 1));
    assert_eq!(fault(&digits, "a"), FAULT_FIELD_TOO_LONG);

    assert_eq!(fault("{\"a\":\"x\ny\"}", "a"), FAULT_FIELD_NOT_TEXT);
    for bytes in [
        &b"{\"a\":\"\xff\"}"[..],
        &b"{\"a\":\"\x80\"}"[..],
        &b"{\"a\":\"\xc3\x28\"}"[..],
        &b"{\"a\":\"\xe6\x9d\"}"[..],
    ] {
        let value = value_at(bytes, "a", DOCUMENT_PAGE);
        assert_eq!(value.fault, FAULT_FIELD_NOT_TEXT);
        assert_eq!(value.value, "");
    }
}

const EVENT_TARGET: &str = "/event/us7000abcd.json";
const COMMENTS_TARGET: &str = "/products/us7000abcd/comments.json";
const COMMENTS: &str =
    r#"{"impact1": "Green alert. There is a low likelihood of damage.", "impact2": ""}"#;

fn name(target: &str) -> DocumentName {
    DocumentName {
        server: "example.org".to_string(),
        target: target.to_string(),
    }
}

fn field(lead: &str, document: u32, path: &str) -> Field {
    Field {
        lead: lead.to_string(),
        document,
        path: steps(path),
    }
}

fn recipe(base: &str) -> Recipe {
    Recipe {
        base: text_chunks(base).into(),
        documents: vec![name(EVENT_TARGET), name(COMMENTS_TARGET)].into(),
        fields: vec![
            field("event_id: ", 0, "id"),
            field("\nmagnitude: ", 0, "properties.mag"),
            field("\nimpact: ", 1, "impact1"),
        ]
        .into(),
        tail: "\n...".to_string(),
    }
}

fn attest(target: &str, document: &[u8]) -> Attestation {
    let mut attestation = Attestation {
        scheme: SCHEME_SECP256K1,
        cipher: CIPHER_NONE,
        time: 1_763_300_000,
        server: "example.org".to_string(),
        target: target.to_string(),
        records_sha256: hex(&sha256(document)),
        records_length: document.len() as u64,
        signature: String::new(),
    };
    sign_attestation(&SECRET, &mut attestation).unwrap();
    attestation
}

fn attested(document: &[u8]) -> (SignerKey, Attestation) {
    (signer_key(&SECRET).unwrap(), attest(EVENT_TARGET, document))
}

fn run(signer: &SignerKey, recipe: &Recipe, given: &[(&Attestation, &str)]) -> SourceRun {
    let sources: Vec<SourceDocument> = given
        .iter()
        .map(|(attestation, body)| SourceDocument {
            attestation,
            body: body.as_bytes(),
        })
        .collect();
    run_source(signer, recipe, &sources)
}

#[test]
fn an_attestation_is_held_to_its_signer_and_every_field() {
    let (signer, attestation) = attested(DOCUMENT.as_bytes());
    assert_eq!(verify_attestation(&signer, &attestation), FAULT_NONE);

    let mut other = attestation.clone();
    other.time += 1;
    assert_eq!(verify_attestation(&signer, &other), FAULT_SIGNATURE);
    let mut other = attestation.clone();
    other.server = "example.com".to_string();
    assert_eq!(verify_attestation(&signer, &other), FAULT_SIGNATURE);
    let mut other = attestation.clone();
    other.target.push('x');
    assert_eq!(verify_attestation(&signer, &other), FAULT_SIGNATURE);
    let mut other = attestation.clone();
    other.records_length += 1;
    assert_eq!(verify_attestation(&signer, &other), FAULT_SIGNATURE);
    let mut other = attestation.clone();
    other.records_sha256 = hex(&sha256(b"another document"));
    assert_eq!(verify_attestation(&signer, &other), FAULT_SIGNATURE);
    let another = signer_key(&[9; 32]).unwrap();
    assert_eq!(verify_attestation(&another, &attestation), FAULT_SIGNATURE);

    let mut other = attestation.clone();
    other.cipher = 1;
    assert_eq!(verify_attestation(&signer, &other), FAULT_ATTESTATION);
    let mut other = attestation.clone();
    other.scheme = 2;
    assert_eq!(verify_attestation(&signer, &other), FAULT_ATTESTATION);
    let mut other = attestation.clone();
    other.signature = "zz".to_string();
    assert_eq!(verify_attestation(&signer, &other), FAULT_ATTESTATION);
    let mut other = attestation.clone();
    other.signature.truncate(10);
    assert_eq!(verify_attestation(&signer, &other), FAULT_ATTESTATION);
    let mut other = attestation.clone();
    other.records_sha256 = "00".to_string();
    assert_eq!(verify_attestation(&signer, &other), FAULT_ATTESTATION);
    let mut unreadable = signer.clone();
    unreadable.key = "02".to_string();
    assert_eq!(
        verify_attestation(&unreadable, &attestation),
        FAULT_ATTESTATION
    );
}

#[test]
fn the_record_states_the_text_the_recipe_and_documents_give() {
    let base = "Decide whether the magnitude is at least 5.0.\n\n";
    let signer = signer_key(&SECRET).unwrap();
    let event = attest(EVENT_TARGET, DOCUMENT.as_bytes());
    let comments = attest(COMMENTS_TARGET, COMMENTS.as_bytes());
    let given = [(&event, DOCUMENT), (&comments, COMMENTS)];
    let made = run(&signer, &recipe(base), &given);
    assert_eq!(
        made.text,
        format!(
            "{base}event_id: \"us7000abcd\"\nmagnitude: 5.1\nimpact: \"Green alert. There is a low likelihood of damage.\"\n..."
        )
    );
    let record = made.record;
    assert!(record.present);
    assert_eq!(record.fault, FAULT_NONE);
    assert_eq!(record.text_sha256, hex(&sha256(made.text.as_bytes())));
    assert_eq!(record.text_total, made.text.len() as u64);
    assert_eq!(record.documents, 2);
    assert_eq!(record.signer, signer.key);

    // The documents' hash is over what each attestation states.
    let mut later = event.clone();
    later.time += 60;
    sign_attestation(&SECRET, &mut later).unwrap();
    let other = run(
        &signer,
        &recipe(base),
        &[(&later, DOCUMENT), (&comments, COMMENTS)],
    )
    .record;
    assert_eq!(other.fault, FAULT_NONE);
    assert_eq!(other.text_sha256, record.text_sha256);
    assert_ne!(other.documents_sha256, record.documents_sha256);
}

#[test]
fn a_source_that_is_wrong_says_so_and_still_finishes() {
    let base = "Base.\n";
    let signer = signer_key(&SECRET).unwrap();
    let event = attest(EVENT_TARGET, DOCUMENT.as_bytes());
    let comments = attest(COMMENTS_TARGET, COMMENTS.as_bytes());
    let good = recipe(base);
    let fault = |recipe: &Recipe, given: &[(&Attestation, &str)]| {
        let made = run(&signer, recipe, given);
        assert_eq!(made.record.text_sha256, hex(&sha256(made.text.as_bytes())));
        (made.record.fault, made.record.fault_position, made.text)
    };

    let altered = DOCUMENT.replace("5.1", "9.9");
    let (code, at, text) = fault(&good, &[(&event, &altered), (&comments, COMMENTS)]);
    assert_eq!((code, at), (FAULT_DOCUMENT, 0));
    assert!(text.contains("magnitude: 9.9"));
    let altered = COMMENTS.replace("low", "high");
    let (code, at, _) = fault(&good, &[(&event, DOCUMENT), (&comments, &altered)]);
    // Which document differs is not kept: the documents are held to the
    // attestations as one chain.
    assert_eq!((code, at), (FAULT_DOCUMENT, 0));
    let (code, _, _) = fault(&good, &[(&event, DOCUMENT), (&comments, "")]);
    assert_eq!(code, FAULT_DOCUMENT);
    let (code, at, _) = fault(&good, &[(&event, &DOCUMENT[1..]), (&comments, COMMENTS)]);
    assert_eq!((code, at), (FAULT_DOCUMENT, 0));

    // Each document is the signer's, and they are not the recipe's.
    let (code, _, _) = fault(&good, &[(&comments, COMMENTS), (&event, DOCUMENT)]);
    assert_eq!(code, FAULT_SERVER);
    let (code, _, text) = fault(&good, &[(&event, DOCUMENT)]);
    assert_eq!(code, FAULT_SERVER);
    assert!(text.ends_with("\nimpact: \n..."));
    let (code, _, _) = fault(
        &good,
        &[
            (&event, DOCUMENT),
            (&comments, COMMENTS),
            (&comments, COMMENTS),
        ],
    );
    assert_eq!(code, FAULT_SERVER);
    let elsewhere = attest("/event/another.json", DOCUMENT.as_bytes());
    let (code, _, _) = fault(&good, &[(&elsewhere, DOCUMENT), (&comments, COMMENTS)]);
    assert_eq!(code, FAULT_SERVER);
    let (code, _, _) = fault(&good, &[]);
    assert_eq!(code, FAULT_SERVER);

    let another = signer_key(&[9; 32]).unwrap();
    let made = run(
        &another,
        &good,
        &[(&event, DOCUMENT), (&comments, COMMENTS)],
    );
    assert_eq!(made.record.fault, FAULT_SIGNATURE);
    assert_eq!(made.record.fault_position, 0);

    let given = [(&event, DOCUMENT), (&comments, COMMENTS)];
    let mut absent = good.clone();
    absent.fields = vec![
        field("a: ", 0, "id"),
        field("\nb: ", 0, "properties.depth"),
        field("\nc: ", 0, "properties"),
    ]
    .into();
    let (code, at, text) = fault(&absent, &given);
    assert_eq!((code, at), (FAULT_FIELD_MISSING, 1));
    assert_eq!(text, "Base.\na: \"us7000abcd\"\nb: \nc: \n...");

    let mut unnamed = good.clone();
    unnamed.fields = vec![field("a: ", 0, "id"), field("\nb: ", 2, "id")].into();
    let (code, at, text) = fault(&unnamed, &given);
    assert_eq!((code, at), (FAULT_RECIPE, 1));
    assert_eq!(text, "Base.\na: \"us7000abcd\"\nb: \n...");

    let mut long = good.clone();
    long.tail = "x".repeat(MAX_PART + 1);
    assert_eq!(fault(&long, &given).0, FAULT_RECIPE);
}

#[test]
fn the_recipe_hash_is_over_what_the_recipe_says() {
    let base = "A base prompt that is long enough to be cut. ".repeat(20);
    let signer = signer_key(&SECRET).unwrap();
    let event = attest(EVENT_TARGET, DOCUMENT.as_bytes());
    let comments = attest(COMMENTS_TARGET, COMMENTS.as_bytes());
    let hash = |recipe: &Recipe| {
        run(
            &signer,
            recipe,
            &[(&event, DOCUMENT), (&comments, COMMENTS)],
        )
        .record
        .recipe_sha256
    };
    let good = recipe(&base);
    assert!(good.base.iter().count() > 1);
    let stated = hash(&good);

    let mut recut = good.clone();
    recut.base = base
        .as_bytes()
        .chunks(7)
        .map(|chunk| String::from_utf8(chunk.to_vec()).unwrap())
        .collect::<Vec<_>>()
        .into();
    assert_eq!(hash(&recut), stated);

    let mut other = good.clone();
    other.base = text_chunks(&format!("{base}!")).into();
    assert_ne!(hash(&other), stated);
    let mut other = good.clone();
    other.tail.push('!');
    assert_ne!(hash(&other), stated);
    let mut other = good.clone();
    other.documents = vec![name(EVENT_TARGET), name("/elsewhere.json")].into();
    assert_ne!(hash(&other), stated);
    let mut other = good.clone();
    other.documents = vec![name(EVENT_TARGET)].into();
    assert_ne!(hash(&other), stated);

    let fields: Vec<Field> = good.fields.iter().cloned().collect();
    let with = |change: &dyn Fn(&mut Vec<Field>)| {
        let mut changed = fields.clone();
        change(&mut changed);
        let mut other = good.clone();
        other.fields = changed.into();
        hash(&other)
    };
    assert_ne!(with(&|f| f[1].path = steps("properties.magnitude")), stated);
    assert_ne!(with(&|f| f[0].lead = "event: ".to_string()), stated);
    assert_ne!(with(&|f| f[2].document = 0), stated);
    assert_ne!(with(&|f| f.swap(0, 1)), stated);
    assert_ne!(
        with(&|f| {
            f.pop();
        }),
        stated
    );
    // Where a lead ends and a path begins is part of the hash.
    assert_ne!(
        with(&|f| {
            f[0].lead = "event_id: i".to_string();
            f[0].path = "d".to_string();
        }),
        stated
    );
}

#[test]
fn text_is_cut_at_characters() {
    let text = "東北 Tōhoku — ".repeat(60);
    let chunks = text_chunks(&text);
    assert!(chunks.len() > 1);
    assert!(chunks.iter().all(|chunk| chunk.len() <= BASE_CHUNK));
    assert_eq!(chunks.concat(), text);
    assert!(text_chunks("").is_empty());
}

#[test]
fn a_step_into_an_array_is_an_index() {
    let document =
        r#"{"list":[10,"two",{"a":[[1,2],[3,{"deep":"x"}]]},[],null],"0":"zero","k":{"1":"one"}}"#;
    assert_eq!(found(document, "list.0"), "10");
    assert_eq!(found(document, "list.1"), "\"two\"");
    assert_eq!(found(document, "list.4"), "null");
    assert_eq!(found(document, "list.2.a.0.1"), "2");
    assert_eq!(found(document, "list.2.a.1.0"), "3");
    assert_eq!(found(document, "list.2.a.1.1.deep"), "\"x\"");
    // In an object a step of digits is a key.
    assert_eq!(found(document, "0"), "\"zero\"");
    assert_eq!(found(document, "k.1"), "\"one\"");

    for path in [
        "list.5", "list.3.0", "list.x", "list.00", "list.01", "list.-1", "list.0.0",
    ] {
        assert_eq!(fault(document, path), FAULT_FIELD_MISSING, "{path}");
    }
    for path in ["list", "list.2", "list.3", "list.2.a.0"] {
        assert_eq!(fault(document, path), FAULT_FIELD_NOT_SCALAR, "{path}");
    }
    for path in ["list.0", "list.2.a.1.1.deep", "list.4", "list.5", "list.3"] {
        let whole = value_at(document.as_bytes(), path, document.len());
        for page in 1..=document.len() {
            assert_eq!(
                value_at(document.as_bytes(), path, page),
                whole,
                "{path} at {page}"
            );
        }
    }
}

#[test]
fn a_key_may_hold_a_dot_or_a_slash() {
    let document = r#"{"contents":{"comments.json":{"url":"a"},"json/alerts.json":{"url":"b"}}}"#;
    let value = |path: &str| {
        let mut scan = begin_field_scan(path);
        scan = scan_field_page(scan, document.as_bytes(), path);
        finish_field_scan(scan)
    };
    assert_eq!(value("contents\ncomments.json\nurl").value, "\"a\"");
    assert_eq!(value("contents\njson/alerts.json\nurl").value, "\"b\"");
    assert_eq!(
        value("contents\ncomments\njson\nurl").fault,
        FAULT_FIELD_MISSING
    );
}

#[test]
fn documents_are_held_to_their_attestations_chunk_by_chunk() {
    let signer = signer_key(&SECRET).unwrap();
    let event = attest(EVENT_TARGET, DOCUMENT.as_bytes());
    let empty = attest(COMMENTS_TARGET, b"");
    let sources = [
        SourceDocument {
            attestation: &event,
            body: DOCUMENT.as_bytes(),
        },
        SourceDocument {
            attestation: &empty,
            body: b"",
        },
    ];
    let good = source_documents(&sources);
    // The empty document still has its one, last chunk.
    assert_eq!(good.chunks.iter().filter(|chunk| chunk.last).count(), 2);
    let mut recipe = recipe("Base.\n");
    recipe.fields = vec![field("id: ", 0, "id")].into();
    let fault =
        |documents: &SourceDocuments| run_source_steps(&signer, &recipe, documents).record.fault;
    assert_eq!(fault(&good), FAULT_NONE);

    let edited = |edit: &dyn Fn(&mut Vec<DocumentChunk>)| {
        let mut chunks: Vec<DocumentChunk> = good.chunks.iter().cloned().collect();
        edit(&mut chunks);
        SourceDocuments {
            attestations: good.attestations.clone(),
            chunks: chunks.into(),
        }
    };
    // A document that never ends, one that ends early, chunks of another
    // document, a missing document and one too many.
    assert_eq!(
        fault(&edited(&|chunks| chunks[0].last = false)),
        FAULT_DOCUMENT
    );
    assert_eq!(
        fault(&edited(&|chunks| {
            chunks.pop();
        })),
        FAULT_DOCUMENT
    );
    assert_eq!(
        fault(&edited(&|chunks| chunks[0].document = 1)),
        FAULT_DOCUMENT
    );
    assert_eq!(fault(&edited(&|chunks| chunks.swap(0, 1))), FAULT_DOCUMENT);
    assert_eq!(
        fault(&edited(&|chunks| {
            let extra = chunks[1].clone();
            chunks.push(DocumentChunk {
                document: 2,
                ..extra
            });
        })),
        FAULT_DOCUMENT
    );
    assert_eq!(fault(&edited(&|chunks| chunks.clear())), FAULT_DOCUMENT);

    // The same bytes cut elsewhere are the same documents.
    let recut = edited(&|chunks| {
        let whole = chunks.remove(0);
        let (head, tail) = whole.bytes.as_slice().split_at(7);
        let piece = |bytes: &[u8], last| DocumentChunk {
            document: 0,
            last,
            bytes: BytesPage::__from_parts(0, 0, bytes.to_vec()),
        };
        chunks.insert(0, piece(tail, true));
        chunks.insert(0, piece(head, false));
    });
    let run = run_source_steps(&signer, &recipe, &recut);
    assert_eq!(run.record.fault, FAULT_NONE);
    assert_eq!(run.text, "Base.\nid: \"us7000abcd\"\n...");
    assert_eq!(run.record, run_source_steps(&signer, &recipe, &good).record);
}
