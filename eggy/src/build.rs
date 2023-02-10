fn main() {
    pyo3_build_config::add_extension_module_link_args();
    println!(
        "cargo:rustc-link-arg=-Wl,-rpath,/Library/Developer/CommandLineTools/Library/Frameworks"
    );
}