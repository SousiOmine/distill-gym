import pytest
from distill_gym.taskgen.contexts import StaticContextProvider, FileContextProvider


class TestStaticContextProvider:
    @pytest.mark.asyncio
    async def test_returns_text(self):
        provider = StaticContextProvider()
        result = await provider.get_context({"text": "hello world"})
        assert result == "hello world"

    @pytest.mark.asyncio
    async def test_no_config_returns_empty(self):
        provider = StaticContextProvider()
        result = await provider.get_context()
        assert result == ""


class TestFileContextProvider:
    @pytest.mark.asyncio
    async def test_no_paths_returns_empty(self):
        provider = FileContextProvider()
        result = await provider.get_context({})
        assert result == ""

    @pytest.mark.asyncio
    async def test_nonexistent_path_returns_empty(self):
        provider = FileContextProvider()
        result = await provider.get_context({"paths": ["/nonexistent/path"]})
        assert result == ""

    @pytest.mark.asyncio
    async def test_reads_single_file(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello", encoding="utf-8")

        provider = FileContextProvider()
        result = await provider.get_context({"paths": [str(test_file)]})
        assert "hello" in result
        assert "test.txt" in result

    @pytest.mark.asyncio
    async def test_reads_multiple_files(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("file a", encoding="utf-8")
        f2.write_text("file b", encoding="utf-8")

        provider = FileContextProvider()
        result = await provider.get_context({"paths": [str(tmp_path)]})
        assert "file a" in result
        assert "file b" in result

    @pytest.mark.asyncio
    async def test_max_files_limit(self, tmp_path):
        for i in range(10):
            (tmp_path / f"f{i}.txt").write_text(f"content {i}", encoding="utf-8")

        provider = FileContextProvider()
        result = await provider.get_context({"paths": [str(tmp_path)], "max_files": 3})
        assert result.count("content") <= 3
