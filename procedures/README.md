# Procedure kinds

One file per procedure kind. The file's name is the kind's name, and what a
specification writes as `procedure_kind`.

A file is the kind's shape: its steps in order. Each step is a use of a
segment kind, and says which earlier step each of its inputs comes from.

```toml
[[step]]
name = "source"          # the step's name; a later step takes from it by this
segment = "source"       # source, inference or result

[[step]]
name = "inference"
segment = "inference"
inputs.source = "source" # the first bind takes the source step's record

[[step]]
name = "result"
segment = "result"
program = "claim-result" # needed where the segment has several programs
inputs.check = "inference"
```

The files are compiled into `iris`. To add a kind, add its file and build;
`cargo test -p procedure-definition` reads every file and refuses one a
pipeline cannot be made of. The segment kinds, their programs and what each
takes are declared in `procedure/definition/src/segment_kind.rs`.

What a shape must hold today:

- its last step is its one result step, named `result`;
- it has at least one inference, and one model per inference in the
  specification, in order;
- it has at most one source step, first and named `source`. A case with no
  documents runs without it.

See `docs/protocol/procedure-kinds.md`.
