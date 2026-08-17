"""Token lifecycle, step-up, and registration-variant flows of the SDK's OAuth client.

Every test connects end to end via `connect_with_oauth`; the assertions are recording-first
(the recorded request sequence is asserted before, or independently of, the call result), so a
surprise in the refresh or step-up paths produces a readable diff of what fired rather than an
opaque failure. The provider knobs that drive each scenario are documented per test.
"""

import base64
from collections import Counter
from urllib.parse import parse_qsl, urlsplit

import anyio
import mcp_types as types
import pytest
from inline_snapshot import snapshot
from mcp_types import INTERNAL_ERROR, ListToolsResult, Tool
from pydantic import AnyHttpUrl, AnyUrl

from mcp import MCPError
from mcp.client.auth import OAuthClientProvider, OAuthTokenError
from mcp.client.auth.extensions.client_credentials import ClientCredentialsOAuthProvider, PrivateKeyJWTOAuthProvider
from mcp.server import Server, ServerRequestContext
from mcp.server.auth.settings import AuthSettings
from mcp.shared.auth import OAuthClientInformationFull, OAuthMetadata
from tests.interaction._connect import BASE_URL
from tests.interaction._requirements import requirement
from tests.interaction.auth._harness import (
    REDIRECT_URI,
    AppShim,
    HeadlessOAuth,
    InMemoryTokenStorage,
    RecordedRequest,
    auth_settings,
    connect_with_oauth,
    m2m_token_shim,
    metadata_body,
    oauth_client_metadata,
    path_prefixed_as_shim,
    record_requests,
    shim,
    step_up_shim,
)
from tests.interaction.auth._provider import InMemoryAuthorizationServerProvider

pytestmark = pytest.mark.anyio

PRM_PATH = "/.well-known/oauth-protected-resource/mcp"
ASM_PATH = "/.well-known/oauth-authorization-server"
CIMD_URL = "https://client.example/.well-known/mcp-client"


async def list_tools(ctx: ServerRequestContext, params: types.PaginatedRequestParams | None) -> ListToolsResult:
    return ListToolsResult(tools=[Tool(name="echo", input_schema={"type": "object"})])


def form_body(request: RecordedRequest) -> dict[str, str]:
    """Parse an `application/x-www-form-urlencoded` request body into a flat dict."""
    return dict(parse_qsl(request.content.decode()))


def authorize_params(authorize_url: str) -> dict[str, str]:
    """Parse the authorize URL's query string into a flat dict."""
    return dict(parse_qsl(urlsplit(authorize_url).query))


def find(recorded: list[RecordedRequest], method: str, path: str) -> list[RecordedRequest]:
    return [r for r in recorded if r.method == method and r.path == path]


def path_counts(recorded: list[RecordedRequest]) -> Counter[tuple[str, str]]:
    return Counter((r.method, r.path) for r in recorded)


def cimd_supported_metadata() -> bytes:
    """AS metadata advertising `client_id_metadata_document_supported: true` (the SDK server never sets it)."""
    metadata = OAuthMetadata(
        issuer=AnyHttpUrl(f"{BASE_URL}/"),
        authorization_endpoint=AnyHttpUrl(f"{BASE_URL}/authorize"),
        token_endpoint=AnyHttpUrl(f"{BASE_URL}/token"),
        registration_endpoint=AnyHttpUrl(f"{BASE_URL}/register"),
        scopes_supported=["mcp"],
        response_types_supported=["code"],
        grant_types_supported=["authorization_code", "refresh_token"],
        code_challenge_methods_supported=["S256"],
        client_id_metadata_document_supported=True,
    )
    return metadata_body(metadata)


def seeded_client(provider: InMemoryAuthorizationServerProvider, **kwargs: object) -> OAuthClientInformationFull:
    """Register a client with the provider and return its info, for pre-registration and CIMD scenarios."""
    base: dict[str, object] = {
        "client_id": "preregistered",
        "token_endpoint_auth_method": "none",
        "redirect_uris": [AnyUrl(REDIRECT_URI)],
        "grant_types": ["authorization_code", "refresh_token"],
        "scope": "mcp",
    }
    base.update(kwargs)
    info = OAuthClientInformationFull.model_validate(base)
    assert info.client_id is not None
    provider.clients[info.client_id] = info
    return info


async def first_process_login(
    provider: InMemoryAuthorizationServerProvider,
    storage: InMemoryTokenStorage,
    *,
    settings: AuthSettings | None = None,
    app_shim: AppShim | None = None,
) -> None:
    """Run one interactive connect so `storage` holds what a first process would leave on disk.

    The restart scenarios below then build a fresh `OAuthClientProvider` over the same storage,
    which is exactly what a second process does; nothing is seeded by hand, so the registration
    and tokens carry whatever the SDK really persists (issuer stamp, secret expiry, scope).
    """
    server = Server("guarded", on_list_tools=list_tools)
    async with connect_with_oauth(server, provider=provider, storage=storage, settings=settings, app_shim=app_shim) as (
        client,
        _,
    ):
        await client.list_tools()
    assert storage.tokens is not None and storage.tokens.refresh_token is not None
    assert storage.client_info is not None and storage.client_info.issuer is not None


def restarted_provider(storage: InMemoryTokenStorage, headless: HeadlessOAuth | None = None) -> OAuthClientProvider:
    """A provider as a second process would construct it: same storage, fresh in-memory state.

    With `headless=None` no redirect or callback handler is wired, so reaching the interactive
    step raises rather than silently opening a browser the scenario says is unavailable.
    """
    return OAuthClientProvider(
        server_url=f"{BASE_URL}/mcp",
        client_metadata=oauth_client_metadata(),
        storage=storage,
        redirect_handler=headless.redirect_handler if headless is not None else None,
        callback_handler=headless.callback_handler if headless is not None else None,
    )


@requirement("client-auth:refresh:transparent")
async def test_an_expired_access_token_is_transparently_refreshed_before_the_next_request() -> None:
    """An access token the client considers expired is refreshed and the new bearer is used.

    The provider tells the client `expires_in=-3600` for the first token while keeping the
    server-side `expires_at` in the future, so the connect's retry succeeds and the next
    request finds the token expired and refreshes. The recorded requests prove exactly one
    `grant_type=refresh_token` exchange carrying the resource indicator, and the bearer used
    after the refresh is the second access token, which is the one persisted to storage.
    """
    recorded, on_request = record_requests()
    provider = InMemoryAuthorizationServerProvider(issue_expired_first=True)
    storage = InMemoryTokenStorage()
    server = Server("guarded", on_list_tools=list_tools)

    with anyio.fail_after(5):
        async with connect_with_oauth(server, provider=provider, storage=storage, on_request=on_request) as (client, _):
            result = await client.list_tools()

    assert result.tools[0].name == "echo"

    token_posts = find(recorded, "POST", "/token")
    bodies = [form_body(r) for r in token_posts]
    assert [b["grant_type"] for b in bodies] == snapshot(["authorization_code", "refresh_token"])

    refresh_body = bodies[1]
    assert sorted(refresh_body) == snapshot(["client_id", "client_secret", "grant_type", "refresh_token", "resource"])
    assert refresh_body["refresh_token"].startswith("refresh_")
    assert refresh_body["resource"].startswith(BASE_URL)

    bearers = {r.headers["authorization"] for r in recorded if r.path == "/mcp" and "authorization" in r.headers}
    assert len(bearers) == 2
    assert storage.tokens is not None
    assert f"Bearer {storage.tokens.access_token}" in bearers
    assert storage.tokens.expires_in == 3600


@requirement("client-auth:403-scope-upgrade")
async def test_a_403_insufficient_scope_triggers_one_reauthorize_with_the_challenged_scope() -> None:
    """A 403 `insufficient_scope` challenge is answered by one re-authorize with the challenge's scope.

    The shim 403s the second authenticated `/mcp` POST (the `notifications/initialized` request,
    which reaches the auth flow's step-up handler; the first authenticated POST is the post-401
    retry, after which the generator ends without inspecting the response). The challenge names a
    wider scope; step-up reuses cached metadata and the existing client registration,
    re-authorizes with the new scope, and the connect completes. The client is pre-registered
    with both scopes so the server's authorize handler accepts the wider second request. One
    re-authorize, one retry; the spec's SHOULD-retry-limit ("a few") is not enforced.
    """
    recorded, on_request = record_requests()
    provider = InMemoryAuthorizationServerProvider()
    storage = InMemoryTokenStorage(client_info=seeded_client(provider, scope="mcp write"))
    server = Server("guarded", on_list_tools=list_tools)
    settings = auth_settings(required_scopes=["mcp"], valid_scopes=["mcp", "write"])
    challenge = 'Bearer error="insufficient_scope", scope="mcp write"'

    with anyio.fail_after(5):
        async with connect_with_oauth(
            server,
            provider=provider,
            storage=storage,
            settings=settings,
            app_shim=step_up_shim(challenge),
            on_request=on_request,
        ) as (client, headless):
            result = await client.list_tools()

    assert result.tools[0].name == "echo"

    assert len(headless.authorize_urls) == 2
    assert authorize_params(headless.authorize_urls[0])["scope"] == "mcp"
    assert authorize_params(headless.authorize_urls[1])["scope"] == "mcp write"

    counts = path_counts(recorded)
    assert counts[("GET", PRM_PATH)] == 1
    assert counts[("GET", ASM_PATH)] == 1
    assert counts[("POST", "/register")] == 0
    assert counts[("GET", "/authorize")] == 2
    assert counts[("POST", "/token")] == 2


@requirement("client-auth:403-scope-union")
async def test_a_403_step_up_re_authorizes_with_the_union_of_prior_and_challenged_scopes() -> None:
    """The step-up re-authorize requests the union of the previously requested and challenged scopes.

    The first authorization requests `mcp`; the 403 challenges a disjoint `write` (not naming
    `mcp`). Per SEP-2350 the client must re-authorize with `mcp write`, not drop `mcp`. The client
    is pre-registered with both scopes so the server's authorize handler accepts the wider request.
    """
    provider = InMemoryAuthorizationServerProvider()
    storage = InMemoryTokenStorage(client_info=seeded_client(provider, scope="mcp write"))
    server = Server("guarded", on_list_tools=list_tools)
    settings = auth_settings(required_scopes=["mcp"], valid_scopes=["mcp", "write"])
    challenge = 'Bearer error="insufficient_scope", scope="write"'

    with anyio.fail_after(5):
        async with connect_with_oauth(
            server,
            provider=provider,
            storage=storage,
            settings=settings,
            app_shim=step_up_shim(challenge),
        ) as (client, headless):
            await client.list_tools()

    assert len(headless.authorize_urls) == 2
    assert authorize_params(headless.authorize_urls[0])["scope"] == "mcp"
    assert authorize_params(headless.authorize_urls[1])["scope"] == "mcp write"


@requirement("client-auth:as-binding")
async def test_credentials_bound_to_a_different_issuer_are_discarded_and_the_client_re_registers() -> None:
    """Credentials bound to a stale issuer are dropped and re-registered against the current AS.

    The stored client is bound (SEP-2352) to a different issuer than the one the server's PRM
    advertises, simulating an authorization-server migration. The client must discard it, perform
    Dynamic Client Registration with the current AS, and never present the stale `client_id` at the
    authorize or token endpoints.
    """
    recorded, on_request = record_requests()
    provider = InMemoryAuthorizationServerProvider()
    stale = seeded_client(provider, client_id="stale-as-client", issuer="https://old-as.example.com")
    storage = InMemoryTokenStorage(client_info=stale)
    server = Server("guarded", on_list_tools=list_tools)

    with anyio.fail_after(5):
        async with connect_with_oauth(server, provider=provider, storage=storage, on_request=on_request) as (
            client,
            _,
        ):
            await client.list_tools()

    # The client re-registered with the current AS...
    assert path_counts(recorded)[("POST", "/register")] == 1
    # ...and the stale client_id never reached the authorize or token endpoints.
    authorize_and_token = find(recorded, "GET", "/authorize") + find(recorded, "POST", "/token")
    assert all("stale-as-client" not in r.url.query.decode() for r in authorize_and_token)
    assert all("stale-as-client" not in r.content.decode() for r in find(recorded, "POST", "/token"))
    # The persisted client is now bound to the current AS.
    assert storage.client_info is not None
    assert storage.client_info.client_id != "stale-as-client"
    assert storage.client_info.issuer == f"{BASE_URL}/"


@requirement("client-auth:401-after-auth-throws")
async def test_a_second_401_after_a_completed_oauth_flow_surfaces_without_looping() -> None:
    """A 401 on the post-auth retry surfaces as an error rather than re-entering discovery.

    The provider rejects every token at verification, so the full flow runs once and the retry
    is 401'd. The auth-flow generator ends after that retry, so the 401 propagates and the
    transport converts it to an INTERNAL_ERROR result, raising during connect. Discovery,
    registration, authorize, and token each ran exactly once: no loop.
    """
    recorded, on_request = record_requests()
    provider = InMemoryAuthorizationServerProvider(reject_all_tokens=True)
    server = Server("guarded", on_list_tools=list_tools)

    def is_internal_error(error: MCPError) -> bool:
        return error.error.code == INTERNAL_ERROR

    with anyio.fail_after(5):
        with pytest.RaisesGroup(pytest.RaisesExc(MCPError, check=is_internal_error), flatten_subgroups=True):
            # Entering the connect raises during the OAuth handshake (inside `Client.__aenter__`),
            # so an `async with` body would be unreachable; entering explicitly avoids dead code.
            await connect_with_oauth(server, provider=provider, on_request=on_request).__aenter__()

    counts = path_counts(recorded)
    assert counts[("GET", PRM_PATH)] == 1
    assert counts[("GET", ASM_PATH)] == 1
    assert counts[("POST", "/register")] == 1
    assert counts[("GET", "/authorize")] == 1
    assert counts[("POST", "/token")] == 1
    assert counts[("POST", "/mcp")] == 2


@requirement("client-auth:cimd")
async def test_cimd_is_selected_when_the_as_advertises_support_and_a_metadata_url_is_supplied() -> None:
    """A client-ID metadata-document URL is used as `client_id` instead of registering.

    AS metadata is shimmed to advertise `client_id_metadata_document_supported: true`; the
    provider is pre-seeded so the server's authorize and token handlers accept the URL as a
    client_id (the SDK server has no CIMD-aware client lookup of its own). The recorded
    requests prove no `/register` call, the authorize URL's `client_id` is the CIMD URL, the
    token request uses `token_endpoint_auth_method=none`, and storage persists the URL as
    `client_id`.
    """
    recorded, on_request = record_requests()
    provider = InMemoryAuthorizationServerProvider()
    seeded_client(provider, client_id=CIMD_URL)
    storage = InMemoryTokenStorage()
    server = Server("guarded", on_list_tools=list_tools)

    with anyio.fail_after(5):
        async with connect_with_oauth(
            server,
            provider=provider,
            storage=storage,
            client_metadata_url=CIMD_URL,
            app_shim=shim(serve={ASM_PATH: cimd_supported_metadata()}),
            on_request=on_request,
        ) as (client, headless):
            await client.list_tools()

    assert find(recorded, "POST", "/register") == []
    assert headless.authorize_url is not None
    assert authorize_params(headless.authorize_url)["client_id"] == CIMD_URL

    [token_req] = find(recorded, "POST", "/token")
    body = form_body(token_req)
    assert body["client_id"] == CIMD_URL
    assert "client_secret" not in body
    assert "authorization" not in token_req.headers

    assert storage.client_info is not None
    assert storage.client_info.client_id == CIMD_URL
    assert storage.client_info.token_endpoint_auth_method == "none"


@requirement("client-auth:invalid-grant-clears-tokens")
async def test_a_failed_refresh_clears_stored_tokens_and_restarts_the_full_flow() -> None:
    """A non-200 refresh response clears the in-memory tokens and the flow re-runs from discovery.

    The first token is reported expired so the next request refreshes; the provider denies the
    refresh once with `invalid_grant`, the auth flow clears its tokens, the unauthenticated
    request 401s, and discovery, authorize, and token run again. The original registration is
    preserved (`client_info` is not cleared). The SDK clears tokens on any non-200 refresh
    response, not specifically `error=invalid_grant`; `source="sdk"` so this is a precision
    note rather than a divergence.
    """
    recorded, on_request = record_requests()
    provider = InMemoryAuthorizationServerProvider(issue_expired_first=True, fail_next_refresh=True)
    storage = InMemoryTokenStorage()
    server = Server("guarded", on_list_tools=list_tools)

    with anyio.fail_after(5):
        async with connect_with_oauth(server, provider=provider, storage=storage, on_request=on_request) as (client, _):
            result = await client.list_tools()

    assert result.tools[0].name == "echo"

    token_posts = find(recorded, "POST", "/token")
    assert [form_body(r)["grant_type"] for r in token_posts] == snapshot(
        ["authorization_code", "refresh_token", "authorization_code"]
    )

    counts = path_counts(recorded)
    assert counts[("POST", "/register")] == 1
    assert counts[("GET", "/authorize")] == 2
    assert counts[("GET", PRM_PATH)] == 2
    assert counts[("GET", ASM_PATH)] == 2

    assert storage.client_info is not None
    assert storage.tokens is not None
    assert storage.tokens.access_token in provider.access_tokens


@requirement("client-auth:refresh:on-401")
async def test_a_restarted_client_answers_a_401_with_its_stored_refresh_token() -> None:
    """A second process holding only persisted tokens and registration refreshes on 401 instead of re-authorizing.

    Steps: (1) a first process logs in interactively and its storage keeps the registration and a
    refresh token; (2) the server-side access token lapses; (3) a fresh provider over the same
    storage, with no browser available, connects. The recording proves the stale bearer drew a
    401, discovery ran, exactly one `refresh_token` grant followed, and neither `/authorize` nor
    `/register` was touched. SDK behaviour per RFC 6749 §1.5; regression bar for #3250/#1318.
    """
    provider = InMemoryAuthorizationServerProvider()
    storage = InMemoryTokenStorage()
    with anyio.fail_after(5):
        await first_process_login(provider, storage)
    assert storage.tokens is not None
    stale_access_token = storage.tokens.access_token
    provider.expire_access_token(stale_access_token)

    recorded, on_request = record_requests()
    server = Server("guarded", on_list_tools=list_tools)
    with anyio.fail_after(5):
        async with connect_with_oauth(
            server, provider=provider, auth=restarted_provider(storage), on_request=on_request
        ) as (client, _):
            result = await client.list_tools()

    assert result.tools[0].name == "echo"
    assert [(r.method, r.path) for r in recorded[:5]] == snapshot(
        [
            ("POST", "/mcp"),
            ("GET", "/.well-known/oauth-protected-resource/mcp"),
            ("GET", "/.well-known/oauth-authorization-server"),
            ("POST", "/token"),
            ("POST", "/mcp"),
        ]
    )
    assert recorded[0].headers["authorization"] == f"Bearer {stale_access_token}"
    assert [form_body(r)["grant_type"] for r in find(recorded, "POST", "/token")] == ["refresh_token"]
    assert find(recorded, "GET", "/authorize") == [] and find(recorded, "POST", "/register") == []
    assert storage.tokens.access_token != stale_access_token
    assert storage.tokens.access_token in provider.access_tokens


@requirement("client-auth:refresh:discovered-endpoint")
async def test_a_refresh_before_the_first_request_discovers_metadata_and_posts_to_the_advertised_token_endpoint() -> (
    None
):
    """A cold-start refresh against an authorization server under a path targets the metadata's token endpoint.

    The authorization server's endpoints live under `/oauth2/v1` and the bare `/token` 404s. The
    second process's storage reports the loaded token as already expired, so the provider
    refreshes before sending anything; the recording proves discovery ran first and the single
    refresh POST went to `/oauth2/v1/token`, with no 401 and no browser. Regression bar for #3240.
    """
    prefix = "/oauth2/v1"
    provider = InMemoryAuthorizationServerProvider(issuer=f"{BASE_URL}{prefix}")
    storage = InMemoryTokenStorage()
    app_shim = path_prefixed_as_shim(prefix)
    with anyio.fail_after(5):
        await first_process_login(provider, storage, app_shim=app_shim)
    storage.report_expired_on_load = True

    recorded, on_request = record_requests()
    server = Server("guarded", on_list_tools=list_tools)
    with anyio.fail_after(5):
        async with connect_with_oauth(
            server, provider=provider, auth=restarted_provider(storage), app_shim=app_shim, on_request=on_request
        ) as (client, _):
            result = await client.list_tools()

    assert result.tools[0].name == "echo"
    assert [(r.method, r.path) for r in recorded[:4]] == snapshot(
        [
            ("GET", "/.well-known/oauth-protected-resource/mcp"),
            ("GET", "/.well-known/oauth-authorization-server/oauth2/v1"),
            ("POST", "/oauth2/v1/token"),
            ("POST", "/mcp"),
        ]
    )
    token_posts = [r for r in recorded if r.method == "POST" and r.path.endswith("/token")]
    assert [(r.path, form_body(r)["grant_type"]) for r in token_posts] == [("/oauth2/v1/token", "refresh_token")]
    assert all("authorization" in r.headers for r in recorded if r.path == "/mcp")
    assert not any(r.path.endswith("/authorize") for r in recorded)


@requirement("client-auth:registration:secret-expiry")
async def test_a_stored_registration_with_a_lapsed_secret_is_replaced_before_authorizing() -> None:
    """A registration whose RFC 7591 `client_secret_expires_at` has passed is discarded and re-made before any consent.

    Steps: (1) first process registers (the server issues secrets expiring in an hour) and logs
    in; (2) the secret's expiry passes on both sides and the storage reports the access token as
    expired too; (3) a second process with a browser connects. The cold-start pass discovers,
    finds the registration unusable and drops it (with its tokens) without registering
    unprompted; the now-unauthenticated request draws the 401 whose flow registers a new client
    and authorizes once. The dead client_id never reaches `/authorize` or `/token`.
    Regression bar for #3256's proactive half.
    """
    settings = auth_settings(client_secret_expiry_seconds=3600)
    provider = InMemoryAuthorizationServerProvider()
    storage = InMemoryTokenStorage()
    with anyio.fail_after(5):
        await first_process_login(provider, storage, settings=settings)
    assert storage.client_info is not None
    dead_client_id = storage.client_info.client_id
    provider.lapse_client_secret(dead_client_id)
    storage.client_info = provider.clients[dead_client_id].model_copy(update={"issuer": storage.client_info.issuer})
    storage.report_expired_on_load = True

    recorded, on_request = record_requests()
    headless = HeadlessOAuth()
    server = Server("guarded", on_list_tools=list_tools)
    with anyio.fail_after(5):
        async with connect_with_oauth(
            server,
            provider=provider,
            settings=settings,
            auth=restarted_provider(storage, headless),
            headless=headless,
            on_request=on_request,
        ) as (client, _):
            result = await client.list_tools()

    assert result.tools[0].name == "echo"
    assert [(r.method, r.path) for r in recorded[:8]] == snapshot(
        [
            ("GET", "/.well-known/oauth-protected-resource/mcp"),
            ("GET", "/.well-known/oauth-authorization-server"),
            ("POST", "/mcp"),
            ("GET", "/.well-known/oauth-protected-resource/mcp"),
            ("GET", "/.well-known/oauth-authorization-server"),
            ("POST", "/register"),
            ("GET", "/authorize"),
            ("POST", "/token"),
        ]
    )
    assert "authorization" not in recorded[2].headers
    assert storage.client_info.client_id != dead_client_id
    assert authorize_params(headless.authorize_urls[0])["client_id"] == storage.client_info.client_id
    assert len(headless.authorize_urls) == 1
    assert all(form_body(r).get("client_id") != dead_client_id for r in find(recorded, "POST", "/token"))


@requirement("client-auth:invalid-client-clears-all")
async def test_an_invalid_client_on_refresh_discards_the_registration_and_reregisters() -> None:
    """When only the server knows the registration is dead, `invalid_client` on refresh triggers one re-registration.

    The stored record still claims a live secret, so nothing is dropped proactively; the 401 flow
    refreshes, the token endpoint answers 401 `invalid_client`, and the flow discards the
    registration with its tokens, registers again, and authorizes with the new client. Reactive
    half of #3256; the TypeScript SDK's `invalidateCredentials('client')` equivalent.
    """
    settings = auth_settings(client_secret_expiry_seconds=3600)
    provider = InMemoryAuthorizationServerProvider()
    storage = InMemoryTokenStorage()
    with anyio.fail_after(5):
        await first_process_login(provider, storage, settings=settings)
    assert storage.client_info is not None and storage.tokens is not None
    dead_client_id = storage.client_info.client_id
    provider.lapse_client_secret(dead_client_id)
    provider.expire_access_token(storage.tokens.access_token)

    recorded, on_request = record_requests()
    headless = HeadlessOAuth()
    server = Server("guarded", on_list_tools=list_tools)
    with anyio.fail_after(5):
        async with connect_with_oauth(
            server,
            provider=provider,
            settings=settings,
            auth=restarted_provider(storage, headless),
            headless=headless,
            on_request=on_request,
        ) as (client, _):
            result = await client.list_tools()

    assert result.tools[0].name == "echo"
    assert [(r.method, r.path) for r in recorded[:7]] == snapshot(
        [
            ("POST", "/mcp"),
            ("GET", "/.well-known/oauth-protected-resource/mcp"),
            ("GET", "/.well-known/oauth-authorization-server"),
            ("POST", "/token"),
            ("POST", "/register"),
            ("GET", "/authorize"),
            ("POST", "/token"),
        ]
    )
    token_posts = find(recorded, "POST", "/token")
    assert [(form_body(r)["grant_type"], form_body(r)["client_id"] == dead_client_id) for r in token_posts] == snapshot(
        [("refresh_token", True), ("authorization_code", False)]
    )
    assert storage.client_info.client_id != dead_client_id
    assert len(headless.authorize_urls) == 1


@requirement("client-auth:invalid-client-clears-all")
async def test_an_invalid_client_at_the_code_exchange_reregisters_and_authorizes_once_more() -> None:
    """A dead registration revealed only at the code exchange is replaced and the flow authorizes once more.

    The second process holds the SDK-minted registration but no tokens, and the server has
    expired the secret without the stored record saying so. The first consent's exchange fails
    `invalid_client`; the flow discards the registration, registers again, and authorizes a
    second time, which succeeds. Two `/authorize` visits is the price of an undeclared expiry.
    """
    settings = auth_settings(client_secret_expiry_seconds=3600)
    provider = InMemoryAuthorizationServerProvider()
    storage = InMemoryTokenStorage()
    with anyio.fail_after(5):
        await first_process_login(provider, storage, settings=settings)
    assert storage.client_info is not None
    dead_client_id = storage.client_info.client_id
    provider.lapse_client_secret(dead_client_id)
    storage.tokens = None

    recorded, on_request = record_requests()
    headless = HeadlessOAuth()
    server = Server("guarded", on_list_tools=list_tools)
    with anyio.fail_after(5):
        async with connect_with_oauth(
            server,
            provider=provider,
            settings=settings,
            auth=restarted_provider(storage, headless),
            headless=headless,
            on_request=on_request,
        ) as (client, _):
            result = await client.list_tools()

    assert result.tools[0].name == "echo"
    assert [(r.method, r.path) for r in recorded[:8]] == snapshot(
        [
            ("POST", "/mcp"),
            ("GET", "/.well-known/oauth-protected-resource/mcp"),
            ("GET", "/.well-known/oauth-authorization-server"),
            ("GET", "/authorize"),
            ("POST", "/token"),
            ("POST", "/register"),
            ("GET", "/authorize"),
            ("POST", "/token"),
        ]
    )
    assert [authorize_params(u)["client_id"] == dead_client_id for u in headless.authorize_urls] == [True, False]
    assert storage.client_info.client_id != dead_client_id
    assert storage.tokens is not None and storage.tokens.access_token in provider.access_tokens


@requirement("client-auth:invalid-client-clears-all")
async def test_an_invalid_client_for_pre_registered_credentials_surfaces_as_a_token_error() -> None:
    """A registration the application supplied is never swapped for a dynamic one; `invalid_client` surfaces.

    The stored client info carries no SDK issuer stamp (it was pre-registered), so when the
    token endpoint rejects its lapsed secret the flow raises `OAuthTokenError` rather than
    registering behind the operator's back. The recording proves no `/register` was attempted.
    """
    provider = InMemoryAuthorizationServerProvider()
    info = seeded_client(
        provider,
        client_id="preregistered",
        client_secret="issued-out-of-band",
        token_endpoint_auth_method="client_secret_post",
    )
    provider.lapse_client_secret("preregistered")
    storage = InMemoryTokenStorage(client_info=info)
    recorded, on_request = record_requests()
    server = Server("guarded", on_list_tools=list_tools)

    with anyio.fail_after(5):
        with pytest.RaisesGroup(pytest.RaisesExc(OAuthTokenError), flatten_subgroups=True):
            await connect_with_oauth(server, provider=provider, storage=storage, on_request=on_request).__aenter__()

    counts = path_counts(recorded)
    assert counts[("POST", "/register")] == 0
    assert counts[("GET", "/authorize")] == 1
    assert counts[("POST", "/token")] == 1
    assert storage.client_info is info


@requirement("client-auth:client-credentials")
async def test_client_credentials_provider_obtains_a_token_without_an_authorize_step() -> None:
    """The client-credentials provider connects with no authorize step and a `client_credentials` grant.

    The SDK server's `TokenHandler` does not route `client_credentials`, so the harness shim
    handles it (the shim is harness; the SDK-under-test is the client provider). The recorded
    `/token` body proves the grant type, scope, resource indicator, and HTTP-Basic client
    authentication; no `/authorize` or `/register` request was made.
    """
    recorded, on_request = record_requests()
    provider = InMemoryAuthorizationServerProvider()
    server = Server("guarded", on_list_tools=list_tools)

    auth = ClientCredentialsOAuthProvider(
        server_url=f"{BASE_URL}/mcp",
        storage=InMemoryTokenStorage(),
        client_id="m2m-client",
        client_secret="m2m-secret",
        scope="mcp",
    )

    with anyio.fail_after(5):
        async with connect_with_oauth(
            server,
            provider=provider,
            auth=auth,
            app_shim=m2m_token_shim(provider, scopes=["mcp"]),
            on_request=on_request,
        ) as (client, headless):
            result = await client.list_tools()

    assert result.tools[0].name == "echo"
    assert headless.authorize_url is None
    assert find(recorded, "GET", "/authorize") == []
    assert find(recorded, "POST", "/register") == []

    [token_req] = find(recorded, "POST", "/token")
    body = form_body(token_req)
    assert body == snapshot(
        {"grant_type": "client_credentials", "resource": "http://127.0.0.1:8000/mcp", "scope": "mcp"}
    )
    decoded = base64.b64decode(token_req.headers["authorization"].removeprefix("Basic ")).decode()
    assert decoded == "m2m-client:m2m-secret"


@requirement("client-auth:private-key-jwt")
async def test_private_key_jwt_provider_authenticates_the_token_request_with_an_assertion() -> None:
    """The private-key-JWT provider sends a `client_assertion` on the token request, with the issuer as audience.

    The assertion provider is a closure that records the audience it was called with and returns
    a fixed opaque value (the JWT contents are not the SDK's concern here); the test asserts the
    `client_assertion`/`client_assertion_type` form fields and that the audience matches the AS
    metadata's issuer.
    """
    recorded, on_request = record_requests()
    provider = InMemoryAuthorizationServerProvider()
    server = Server("guarded", on_list_tools=list_tools)

    audiences: list[str] = []

    async def assertion_provider(audience: str) -> str:
        audiences.append(audience)
        return "header.payload.sig"

    auth = PrivateKeyJWTOAuthProvider(
        server_url=f"{BASE_URL}/mcp",
        storage=InMemoryTokenStorage(),
        client_id="m2m-jwt-client",
        assertion_provider=assertion_provider,
        scope="mcp",
    )

    with anyio.fail_after(5):
        async with connect_with_oauth(
            server,
            provider=provider,
            auth=auth,
            app_shim=m2m_token_shim(provider, scopes=["mcp"]),
            on_request=on_request,
        ) as (client, _):
            result = await client.list_tools()

    assert result.tools[0].name == "echo"
    assert audiences == [f"{BASE_URL}/"]

    [token_req] = find(recorded, "POST", "/token")
    body = form_body(token_req)
    assert body == snapshot(
        {
            "grant_type": "client_credentials",
            "client_assertion": "header.payload.sig",
            "client_assertion_type": "urn:ietf:params:oauth:client-assertion-type:jwt-bearer",
            "resource": "http://127.0.0.1:8000/mcp",
            "scope": "mcp",
        }
    )
    assert "client_secret" not in body
    assert "authorization" not in token_req.headers


@pytest.mark.parametrize(
    ("case", "preseed_storage", "advertise_cimd"),
    [("cimd_unsupported_falls_through_to_dcr", False, False), ("preregistered_beats_cimd", True, True)],
    ids=["cimd_unsupported_falls_through_to_dcr", "preregistered_beats_cimd"],
)
@requirement("client-auth:cimd")
async def test_registration_priority_prefers_preregistered_then_cimd_then_dcr(
    case: str, preseed_storage: bool, advertise_cimd: bool
) -> None:
    """The client picks pre-registration over CIMD over DCR, falling through when each is unavailable.

    Two priority edges are exercised: with a CIMD URL configured but no AS support, DCR runs and
    the registered `client_id` is used; with a CIMD URL configured and AS support but a
    pre-registered client in storage, the stored `client_id` is used and neither CIMD nor DCR
    runs. (The positive CIMD case and pre-registration over DCR are covered by their own tests.)
    """
    recorded, on_request = record_requests()
    provider = InMemoryAuthorizationServerProvider()
    server = Server("guarded", on_list_tools=list_tools)
    storage = InMemoryTokenStorage()

    expected_client_id: str
    if preseed_storage:
        info = seeded_client(provider)
        storage.client_info = info
        assert info.client_id is not None
        expected_client_id = info.client_id
    else:
        expected_client_id = ""

    app_shim = shim(serve={ASM_PATH: cimd_supported_metadata()}) if advertise_cimd else None

    with anyio.fail_after(5):
        async with connect_with_oauth(
            server,
            provider=provider,
            storage=storage,
            client_metadata_url=CIMD_URL,
            app_shim=app_shim,
            on_request=on_request,
        ) as (client, headless):
            await client.list_tools()

    assert headless.authorize_url is not None
    chosen_client_id = authorize_params(headless.authorize_url)["client_id"]
    assert chosen_client_id != CIMD_URL

    if case == "cimd_unsupported_falls_through_to_dcr":
        assert len(find(recorded, "POST", "/register")) == 1
        assert chosen_client_id in provider.clients
    else:
        assert find(recorded, "POST", "/register") == []
        assert chosen_client_id == expected_client_id
