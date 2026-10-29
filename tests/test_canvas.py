from palimdiff.canvas import MemoryCanvas
from palimdiff.denoisers.tiny import TinyStandInDenoiser


def test_canvas_capacity_and_free_region():
    d = TinyStandInDenoiser()
    c = MemoryCanvas.from_text(d, "hello world", capacity=20)
    assert len(c.token_ids) == 20
    assert c.free_region is not None
    assert c.free_region.length >= 1


def test_kept_spans():
    d = TinyStandInDenoiser()
    c = MemoryCanvas.from_text(d, "a", capacity=10)
    remask = {2, 3, 7}
    spans = c.kept_spans(remask)
    assert spans[0].start == 0
