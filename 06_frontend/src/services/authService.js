const AUTH_MODE = 'demo';
const SESSION_KEY = 'hddui.auth.session';
const SESSION_TTL_MS = 12 * 60 * 60 * 1000;

const DEMO_USER = Object.freeze({
  id: 'demo-user-1',
  name: 'A. Researcher',
  email: 'researcher@hdd-lab.edu',
  role: 'researcher',
});

const DEMO_PASSWORD = 'Demo@123';

const hash = (value) => {
  let h = 5381;
  for (let i = 0; i < value.length; i += 1) {
    h = ((h << 5) + h + value.charCodeAt(i)) >>> 0;
  }
  return String(h);
};

const DEMO_PASSWORD_HASH = hash(DEMO_PASSWORD);

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const encodePayload = (obj) => btoa(unescape(encodeURIComponent(JSON.stringify(obj))));
const decodePayload = (str) => JSON.parse(decodeURIComponent(escape(atob(str))));

const createSession = (remember) => {
  const issuedAt = Date.now();
  const expiresAt = issuedAt + SESSION_TTL_MS;
  const token = encodePayload({ sub: DEMO_USER.id, mode: AUTH_MODE, iat: issuedAt, exp: expiresAt });
  const session = { token, user: DEMO_USER, expiresAt };
  const store = remember ? localStorage : sessionStorage;
  store.setItem(SESSION_KEY, JSON.stringify(session));
  return session;
};

const clearSession = () => {
  localStorage.removeItem(SESSION_KEY);
  sessionStorage.removeItem(SESSION_KEY);
};

export const authService = {
  AUTH_MODE,

  demoCredentials: Object.freeze({ email: DEMO_USER.email, password: DEMO_PASSWORD }),

  async login({ email, password, remember = false }) {
    await delay(650);
    const normalizedEmail = (email || '').trim().toLowerCase();
    if (!normalizedEmail || !password) {
      throw new Error('Email and password are required.');
    }
    if (normalizedEmail !== DEMO_USER.email || hash(password) !== DEMO_PASSWORD_HASH) {
      throw new Error('Invalid email or password. Use the demo credentials shown below.');
    }
    return createSession(!!remember);
  },

  logout() {
    clearSession();
  },

  restoreSession() {
    const raw = localStorage.getItem(SESSION_KEY) || sessionStorage.getItem(SESSION_KEY);
    if (!raw) return null;
    try {
      const session = JSON.parse(raw);
      if (!session?.token || !session?.user || !session?.expiresAt) {
        clearSession();
        return null;
      }
      const payload = decodePayload(session.token);
      if (payload.sub !== session.user.id || payload.exp <= Date.now() || session.expiresAt <= Date.now()) {
        clearSession();
        return null;
      }
      return session;
    } catch {
      clearSession();
      return null;
    }
  },
};

export default authService;
