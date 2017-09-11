///
/// Just a wrapper around a standard panic! invocation. This is used to indicate to the compiler
/// that a certain line cannot execute (and will cause a panic at runtime if it does). Intended
/// usage is after a loop from an iterator which never returns None.
///
pub fn cannot_happen() -> ! {
    panic!("This line cannot be executed, but the compiler doesn't know it");
}
