"""Network-free regression tests for bounded download behavior."""
import io
import unittest

from download_10min import fetch_range


class Response(io.BytesIO):
    def __init__(self, content, status=206, content_range='bytes 10-13/100'):
        super().__init__(content)
        self.status = status
        self.headers = {'Content-Range': content_range}


class RangeTests(unittest.TestCase):
    def test_exact_range(self):
        out = io.BytesIO()
        def opener(request, timeout):
            self.assertEqual(request.get_header('Range'), 'bytes=10-13')
            return Response(b'abcd')
        fetch_range('https://example.invalid/data', 10, 14, out, 100, opener)
        self.assertEqual(out.getvalue(), b'abcd')

    def test_whole_file_response_rejected(self):
        with self.assertRaises(RuntimeError):
            fetch_range('https://example.invalid/data', 10, 14, io.BytesIO(), 100,
                        lambda *a, **k: Response(b'abcd', status=200))

    def test_wrong_range_rejected(self):
        with self.assertRaises(RuntimeError):
            fetch_range('https://example.invalid/data', 10, 14, io.BytesIO(), 100,
                        lambda *a, **k: Response(b'abcd', content_range='bytes 0-3/100'))

    def test_interrupted_response_raises(self):
        with self.assertRaises(IOError):
            fetch_range('https://example.invalid/data', 10, 14, io.BytesIO(), 100,
                        lambda *a, **k: Response(b'ab'))


if __name__ == '__main__':
    unittest.main()
