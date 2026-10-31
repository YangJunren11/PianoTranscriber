"""How right a bar's LMX is, on a real bar (Beethoven, Piano Sonata No. 10, Op. 14 No. 2, from the bar
dataset) read wrong in the ways a recognizer reads bars wrong."""

from __future__ import annotations

from fractions import Fraction

from transcriber.lmx_accuracy import Tally, score_bar, table
from transcriber.lmx_bars import State, notes_of

CONTEXT = "key:fifths:1 time beats:2 beat-type:4 clef:G2 staff:1 clef:G2 staff:2"
BAR = ("measure D5 voice:1 eighth stem:down staff:1 slur:stop chord F5 eighth D6 quarter D6 eighth tied:start "
       "backup half D4 voice:5 16th stem:up staff:2 beam:begin beam:begin slur:start chord F4 16th D5 16th "
       "E4 16th chord G4 16th D5 16th beam:end beam:end F4 16th beam:begin beam:begin chord A4 16th D5 16th "
       "G4 16th chord B4 16th D5 16th beam:end beam:end")
NOTES = 16


def test_the_bar_reads_back_as_its_notes_with_the_key_applied() -> None:
    notes, valid = notes_of(BAR.split(), State.parse(CONTEXT))
    assert valid and len(notes) == NOTES
    assert [(n.staff, n.onset, n.pitch, n.duration) for n in notes[:4]] == [
        (1, 0, "D5", Fraction(1, 2)), (1, 0, "F#5", Fraction(1, 2)), (1, Fraction(1, 2), "D6", 1),
        (1, Fraction(3, 2), "D6", Fraction(1, 2))]
    # Every bar's notes add up: the left hand's sixteenths fill the 2/4 bar
    assert max(n.onset + n.duration for n in notes if n.staff == 2) == 2


def test_a_bar_read_right_scores_full_marks() -> None:
    s = score_bar(BAR, BAR, CONTEXT)
    assert s.exact and s.valid and s.edits == 0 and s.state_right
    assert s.notes == s.found == s.right == s.pitches_right == s.timed_right == NOTES


def test_a_wrong_pitch_costs_one_token_and_one_note() -> None:
    s = score_bar(BAR.replace(" E4 ", " C4 ", 1), BAR, CONTEXT)
    assert s.edits == 1 and not s.exact and s.valid
    assert s.right == s.pitches_right == s.timed_right == NOTES - 1 and s.found == NOTES


def test_a_wrong_length_is_one_note_misread_but_moves_the_next_voice_in_time() -> None:
    # The right hand's last note made a quarter: the right hand runs an eighth over the bar, and the
    # left hand, written after stepping back a half note, starts an eighth late
    read = BAR.replace("D6 eighth tied:start", "D6 quarter tied:start")
    s = score_bar(read, BAR, CONTEXT)
    assert s.edits == 1 and s.right == NOTES - 1 and s.pitches_right == NOTES
    left = [n for n in notes_of(read.split(), State.parse(CONTEXT))[0] if n.staff == 2]
    assert min(n.onset for n in left) == Fraction(1, 2)
    # In time, the right hand's first three notes stay right, and three of the left hand's D5s, which
    # come every eighth, fall where another D5 was
    assert s.timed_right == 6


def test_voices_stems_and_beams_do_not_count_as_music() -> None:
    drawn_otherwise = " ".join(t for t in BAR.replace("voice:5", "voice:2").split()
                               if not t.startswith(("stem:", "beam:")))
    s = score_bar(drawn_otherwise, BAR, CONTEXT)
    assert s.edits > 10 and not s.exact
    assert s.right == s.pitches_right == s.timed_right == s.found == NOTES and s.valid


def test_a_missed_key_change_is_caught_by_the_state_though_no_note_shows_it() -> None:
    truth = BAR.replace("measure", "measure key:fifths:2", 1)  # to D major: no C in this bar
    s = score_bar(BAR, truth, CONTEXT)
    assert s.right == NOTES and not s.state_right and s.edits == 1


def test_a_missed_tie_over_the_barline_is_caught_by_the_state() -> None:
    s = score_bar(BAR.replace(" tied:start", ""), BAR, CONTEXT)
    assert not s.state_right and s.right == NOTES


def test_a_bar_read_right_after_a_key_misread_sounds_wrong() -> None:
    # Its tokens are the true bar's, but the bars before left C major for G major: every F in it is wrong
    s = score_bar(BAR, BAR, CONTEXT, read_from=State.parse(CONTEXT.replace("key:fifths:1", "key:fifths:0")))
    assert s.exact and s.edits == 0 and not s.state_right
    assert s.right == NOTES - 3  # F#5 and the two F#4s


def test_nonsense_and_nothing_find_no_notes_and_do_not_fail() -> None:
    for read in ("measure beam:end tied:stop chord backup", "", "D5 D5 D5 eighth eighth"):
        s = score_bar(read, BAR, CONTEXT)
        assert s.right < 3 and s.notes == NOTES and not s.exact


def test_two_bars_read_as_one_are_not_valid() -> None:
    s = score_bar(BAR + " " + BAR, BAR, CONTEXT)
    assert not s.valid and s.found == 2 * NOTES and s.right == NOTES


def test_scores_add_up_by_tokens_and_notes() -> None:
    t = Tally()
    t.add(score_bar(BAR, BAR, CONTEXT), "leland")
    t.add(score_bar(BAR.replace(" E4 ", " C4 ", 1), BAR, CONTEXT), "bravura")
    s = t.summary()
    assert s["bars"] == 2 and s["bars exact"] == 0.5
    assert s["token error rate"] == 1 / (2 * len(BAR.split()))
    assert s["note F1"] == 2 * (2 * NOTES - 1) / (4 * NOTES)
    assert set(t.by) == {"leland", "bravura"}
    assert "note F1" in table({"model": t})
