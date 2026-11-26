use std::process::ExitCode;

fn main() -> ExitCode {
    match iwebs_cli::run_from_env() {
        Ok(code) => code,
        Err(error) => {
            eprintln!("iwebs: {error:#}");
            ExitCode::from(1)
        }
    }
}
