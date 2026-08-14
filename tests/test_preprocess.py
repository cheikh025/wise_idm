import io
import unittest

from preprocess_videos import _read_at_most


class ShortReadStream(io.BytesIO):
    def read(self, size=-1):
        return super().read(min(size, 3))


class PreprocessTest(unittest.TestCase):
    def test_read_at_most_fills_across_short_pipe_reads(self):
        stream = ShortReadStream(b"abcdefghij")
        self.assertEqual(_read_at_most(stream, 8), b"abcdefgh")
        self.assertEqual(_read_at_most(stream, 8), b"ij")


if __name__ == "__main__":
    unittest.main()
