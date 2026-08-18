# Supporting v1 and v2 at the same time

This page is for authors of published MCP servers and libraries: packages whose users install `mcp` themselves, often without a version pin. When v2.0.0 released, fresh installs of packages declaring `mcp>=1.2.0` began resolving v2 and crashed at import time with `ModuleNotFoundError: No module named 'mcp.server.fastmcp'`, while already-installed users stayed on v1. The scale is not small: a September 2026 sample of registry-listed PyPI servers found roughly a fifth to a third failing to start on a fresh install for exactly this reason (numbers in [#3309](https://github.com/modelcontextprotocol/python-sdk/issues/3309)). Since [#3388](https://github.com/modelcontextprotocol/python-sdk/pull/3388) the old import path raises a descriptive error that points at the migration guide, but the process still dies, so a published package still has to handle both majors itself. If you can simply migrate, follow the [Migration Guide](migration.md). If your package has to work on both majors during the transition, this page shows the supported pattern.

The SDK itself deliberately ships no backwards-compatibility shims: carrying both surfaces inside one SDK would nearly double the public interface (see the discussion in [#3309](https://github.com/modelcontextprotocol/python-sdk/issues/3309)). The recommended approach is a small shim in your own package. The [Anthropic Python SDK uses the same pattern](https://github.com/anthropics/anthropic-sdk-python/commit/8b327c35c35efda455dba5be6aeaea04672eb760).

## The import shim

`FastMCP` was renamed to `MCPServer` and its module moved, so import one or the other:

```python
try:
    # MCP SDK 2.x: FastMCP was renamed to MCPServer and the module moved.
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:
    # MCP SDK 1.x keeps the original path.
    from mcp.server.fastmcp import FastMCP
```

Alias to whichever name you prefer, but pick one and use it consistently; the rest of your code then reads identically on both lines. For code that stays on the decorator-level surface (the constructor, `@mcp.tool()`, `run()`), this shim is typically the only code change dual support needs: the decorator API behaves the same on both majors.

!!! warning "Pass constructor options as keywords"
    The constructor's positional layout differs between the majors: v1 is `FastMCP(name, instructions=None, ...)` while v2 is `MCPServer(name=None, title=None, description=None, instructions=None, ...)`. A positional call like `FastMCP("my-server", "Use these tools to...")` sets `instructions` on v1 but silently sets `title` on v2. Passing everything except the name as keyword arguments works correctly on both.

## The dependency pin

```toml
dependencies = ["mcp>=1.2.0,<3"]
```

Two deliberate choices here:

- **No `<2` cap.** A `<2` cap is the right call while you are *not yet* dual-compatible (the [Migration Guide](migration.md) says the same), but keeping it after adding the shim strands the growing share of users whose resolvers pick v2.
- **A `<3` cap.** The next major will presumably break something again. An explicit cap turns that day into a controlled version bump on your side instead of a surprise crash on your users' side.

## Test both lines in CI

A fresh `pip install` in CI resolves v2, so without extra care the v1 fallback path becomes dead code that only your users execute. Keep it honest with one extra job that forces the v1 line and runs the same test suite:

```yaml
  test-mcp-v1:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install with MCP SDK 1.x
        run: pip install -e ".[dev]" "mcp>=1.2.0,<2"
      - name: Test
        run: pytest
```

If the identical suite passes on both jobs, you have working dual support rather than a merely survivable import.

## What the shim does not cover

The shim only papers over the `FastMCP` rename. Every other breaking change in v2 still applies at runtime when the resolver picks v2: camelCase to snake_case field renames, removed `mcp.types` aliases, the `httpx` to `httpx2` swap, client-side and low-level server changes. If your package touches any of those surfaces, each needs its own guard, or an honest hard requirement on a single major. The [Migration Guide](migration.md) is the complete list of what moved.

## When to drop v1

The v1.x maintenance line receives critical bug fixes and security patches only. Once your users' installed base has moved (per-version download stats are a good signal), retire the transition machinery in one release: remove the fallback import, tighten the pin to `mcp>=2,<3`, delete the extra CI job, and flag the new floor in your changelog as a breaking change.
