## Multi-user chat history persistence

### Goals
- Persist chat history per user/session and model.
- Allow creating/resuming conversations; keep assistant/system turns intact.
- Keep storage small (caps/TTL) and enforce auth scoping.

### Backend (rag-chat)
- [ ] Choose store: SQLite/Postgres (recommended) or Redis for short-lived sessions.
- [ ] Add config env: `CHAT_DB_URL` (or path), `CHAT_HISTORY_TTL`, `CHAT_HISTORY_MAX_MSGS`.
- [ ] Schema (SQL):
  - `conversations(id uuid, user_id text, model text, created_at, updated_at)`
  - `messages(id uuid, conversation_id uuid, role text, content text, created_at, seq int)`
  - Index on `(user_id, updated_at desc)`.
- [ ] Add DAL helpers: create convo, list convos by user, fetch messages (ordered), append messages with seq, prune by max messages/TTL.
- [ ] Update `/chat`:
  - Accept `conversation_id` (optional). Require user id from auth (service token/Access).
  - Load messages for convo (if provided) + append incoming user turn; prepend system prompt; insert retrieved RAG context.
  - Send to Ollama; store assistant reply; return reply + `conversation_id`.
  - Enforce user scoping on all queries.
- [ ] Add endpoints:
  - `GET /conversations` (list for user)
  - `GET /conversations/{id}` (metadata + recent messages)
  - `POST /conversations` (new conversation, optional title/model)
  - `DELETE /conversations/{id}` (clear)
  - Optional: `POST /conversations/{id}/reset` to wipe messages.
- [ ] Add pruning: drop oldest messages beyond `CHAT_HISTORY_MAX_MSGS`; optional TTL cleanup.

### Frontend (UI)
- [ ] Track `conversation_id` client-side.
- [ ] On load: fetch conversation list + latest messages; hydrate bubbles.
- [ ] On send: include `conversation_id`; if absent, create conversation first (or accept `conversation_id` from response).
- [ ] Add dropdown/list to switch conversations; add “New chat” button that resets history and starts a fresh convo.
- [ ] Keep existing local history as render cache but always reconcile with server response.
- [ ] Handle auth errors and empty histories gracefully.

### Auth & security
- [ ] Derive `user_id` from existing auth/session token (e.g., Cloudflare Access/service token).
- [ ] Enforce user scoping in all queries; never trust client-sent user ids.
- [ ] Sanitize inputs; size-limit message content before storing.

### Limits & ops
- [ ] Set sane defaults: `CHAT_HISTORY_MAX_MSGS` (e.g., 50–100), TTL (optional).
- [ ] Add migration/init step for DB.
- [ ] Add lightweight health/check for DB connectivity.

### Testing
- [ ] Unit: DAL CRUD, pruning, ordering, scoping.
- [ ] Integration: create → chat → resume → prune paths; unauthorized access blocked.
- [ ] UI: conversation switch, new chat, send, clear, refresh restores history.

### Nice-to-haves
- [ ] Conversation titles (first user message or user-provided).
- [ ] Export/download conversation as JSON/markdown.
- [ ] Streamed chat support with incremental storage.
