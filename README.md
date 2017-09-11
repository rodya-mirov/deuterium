# deuterium

This project is mainly for source control for myself, not necessarily for outside use.

It contains solutions to problems from a certain puzzle site which:
(a) has asked users not to distribute answers on the internet, but
(b) where answers are easily searchable on the internet.

I figure so long as I don't actually use the name of that site on this repo, it shouldn't be easily
searchable, so there shouldn't be any harm. Especially in light of (b). Honestly, you get people writing
whole blogs about solving these puzzles (and rightly so -- they're often quite interesting!).

Anyway.

This is where I keep my solutions, because I don't want to lose them to a faulty hard drive, and I think
git makes better backups than dropbox. I don't expect this will be any use to outsiders, but at least there's
a readme.

### Instructions for use:

  `cargo run 137`
  
will run problem 137 and print out the answer, as well as how long it took. Replace 137 with any number
you like. If the problem isn't supported it will give helpful failure output and quit. Not intended to be
robust to bad input.

There is a second crate containing useful library functions, but honestly most of the problems are distinct
enough that there's not much useful there. The mark of a good bunch of problems, I say.
