use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

/// Resolve generated chain paths before saving a template or frozen run elsewhere.
pub fn resolve_raster_manifest_paths(template: &str, base: &Path) -> Result<String> {
    let mut lines = template
        .lines()
        .map(|line| rewrite_manifest_line(line, base))
        .collect::<Result<Vec<_>>>()?;
    lines.push(String::new());
    Ok(lines.join("\n"))
}

fn rewrite_manifest_line(line: &str, template_dir: &Path) -> Result<String> {
    let line = rewrite_project_line(line, template_dir)?;
    let line = rewrite_path_assignments(&line, template_dir, "index_path")?;
    rewrite_path_assignments(&line, template_dir, "path")
}

fn rewrite_project_line(line: &str, template_dir: &Path) -> Result<String> {
    let trimmed = line.trim_start();
    let Some(value) = trimmed.strip_prefix("project =") else {
        return Ok(line.to_string());
    };
    let indent = &line[..line.len() - trimmed.len()];
    let project: String = serde_json::from_str(value.trim()).with_context(|| {
        format!(
            "failed to parse chain stage project path from manifest line `{}`",
            line
        )
    })?;
    let project_path = Path::new(&project);
    if project_path.is_absolute() {
        return Ok(line.to_string());
    }
    let absolute_project = template_dir.join(project_path);
    Ok(format!(
        "{}project = {:?}",
        indent,
        absolute_project.to_string_lossy()
    ))
}

fn rewrite_path_assignments(line: &str, template_dir: &Path, key: &str) -> Result<String> {
    let mut output = String::with_capacity(line.len());
    let mut cursor = 0;
    while let Some(relative_pos) = line[cursor..].find(key) {
        let key_start = cursor + relative_pos;
        let Some(assignment) = parse_path_assignment(line, key, key_start)? else {
            output.push_str(&line[cursor..key_start + key.len()]);
            cursor = key_start + key.len();
            continue;
        };

        output.push_str(&line[cursor..assignment.value_start]);
        if assignment.path.is_absolute() {
            output.push_str(assignment.literal);
        } else {
            let absolute_path = template_dir.join(assignment.path);
            output.push_str(&format!("{:?}", absolute_path.to_string_lossy()));
        }
        cursor = assignment.value_end;
    }
    output.push_str(&line[cursor..]);
    Ok(output)
}

struct PathAssignment<'a> {
    value_start: usize,
    value_end: usize,
    literal: &'a str,
    path: PathBuf,
}

fn parse_path_assignment<'a>(
    line: &'a str,
    key: &str,
    key_start: usize,
) -> Result<Option<PathAssignment<'a>>> {
    if key_start > 0 {
        let previous = line.as_bytes()[key_start - 1];
        if previous.is_ascii_alphanumeric() || previous == b'_' {
            return Ok(None);
        }
    }
    let mut cursor = key_start + key.len();
    cursor = skip_ascii_spaces(line, cursor);
    if line.as_bytes().get(cursor) != Some(&b'=') {
        return Ok(None);
    }
    cursor += 1;
    cursor = skip_ascii_spaces(line, cursor);
    if line.as_bytes().get(cursor) != Some(&b'"') {
        return Ok(None);
    }
    let value_start = cursor;
    let value_end = closing_quote_end(line, value_start)?;
    let literal = &line[value_start..value_end];
    let path: String = serde_json::from_str(literal)
        .with_context(|| format!("failed to parse `{key}` path from manifest line `{line}`"))?;
    Ok(Some(PathAssignment {
        value_start,
        value_end,
        literal,
        path: PathBuf::from(path),
    }))
}

fn skip_ascii_spaces(line: &str, mut cursor: usize) -> usize {
    while matches!(line.as_bytes().get(cursor), Some(b' ' | b'\t')) {
        cursor += 1;
    }
    cursor
}

fn closing_quote_end(line: &str, value_start: usize) -> Result<usize> {
    let mut escaped = false;
    for (offset, byte) in line.as_bytes()[value_start + 1..].iter().enumerate() {
        if escaped {
            escaped = false;
            continue;
        }
        match byte {
            b'\\' => escaped = true,
            b'"' => return Ok(value_start + 1 + offset + 1),
            _ => {}
        }
    }
    anyhow::bail!("unterminated string in manifest line `{line}`")
}
