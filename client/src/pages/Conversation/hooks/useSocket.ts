import { useState, useEffect, useCallback, useRef } from "react";
import { WSMessage, SocketStatus } from "../../../protocol/types";
import { decodeMessage, encodeMessage } from "../../../protocol/encoder";

// How long with zero WS messages (of any kind, including audio) before we consider the
// connection dead and worth cycling. Audio keeps streaming continuously in normal operation,
// so this should only ever fire on a genuinely stuck connection -- but it's kept generous
// (rather than the previous 10s) because a fresh, healthy connection can have a real quiet
// stretch (Opus emits little/no data during silence) and because the server now sends its
// own WS-level ping every 15s (see `heartbeat=15` in server.py), which the browser answers
// automatically without that traffic ever reaching `onMessageEvent`. This timer is a backstop
// for the case where the browser's own close/error events don't fire promptly, not the
// primary liveness signal.
const INACTIVITY_TIMEOUT_MS = 30000;
const INACTIVITY_CHECK_INTERVAL_MS = 1000;

// Reconnect backoff schedule for unexpected disconnects (network blips, an intermediary
// proxy dropping a long-lived connection, a transient server hiccup, etc). Capped so a
// truly-gone server doesn't get hammered forever, but generous enough to ride out anything
// short of the operator intentionally stopping the server.
const RECONNECT_BASE_DELAY_MS = 1000;
const RECONNECT_MAX_DELAY_MS = 10000;
const RECONNECT_MAX_ATTEMPTS = 30;

export const useSocket = ({
  onMessage,
  uri,
  onDisconnect: onDisconnectProp,
  onReconnectExhausted,
}: {
  onMessage?: (message: WSMessage) => void;
  uri: string;
  onDisconnect?: () => void;
  onReconnectExhausted?: () => void;
}) => {
  const lastMessageTime = useRef<null|number>(null);
  const socketRef = useRef<WebSocket | null>(null); // useRef to keep stable socket reference
  const [socketStatus, setSocketStatus] = useState<SocketStatus>("disconnected");
  const manualCloseRef = useRef<boolean>(false);
  const reconnectAttemptsRef = useRef<number>(0);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const sendMessage = useCallback(
    (message: WSMessage) => {
      if (!socketRef.current || socketStatus !== "connected") {
        console.log("socket not connected");
        return;
      }
      socketRef.current.send(encodeMessage(message));
    },
    [socketRef, socketStatus],
  );

  const onConnect = useCallback(() => {
    console.log("connected, now waiting for handshake.");
    setSocketStatus("connecting");
  }, [setSocketStatus]);

  // Forward-declared so onDisconnect (defined before start/connect) can schedule a retry.
  const connectRef = useRef<() => void>(() => {});

  const clearReconnectTimeout = useCallback(() => {
    if (reconnectTimeoutRef.current !== null) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
  }, []);

  const onDisconnect = useCallback((event: CloseEvent) => {
    const closedSocket = event.target as WebSocket;
    console.log("disconnected");
    // ONLY act if it's the current socket that closed (ignore stale sockets from a previous
    // connection attempt that raced with a newer one).
    if (socketRef.current !== closedSocket) {
      console.log("disconnected (stale socket ignored)");
      return;
    }
    socketRef.current = null;
    onDisconnectProp?.();

    if (manualCloseRef.current) {
      // User (or the app) intentionally closed this socket -- don't reconnect.
      setSocketStatus("disconnected");
      return;
    }

    if (reconnectAttemptsRef.current >= RECONNECT_MAX_ATTEMPTS) {
      console.log("giving up on reconnecting after", reconnectAttemptsRef.current, "attempts");
      setSocketStatus("disconnected");
      onReconnectExhausted?.();
      return;
    }

    const attempt = reconnectAttemptsRef.current + 1;
    reconnectAttemptsRef.current = attempt;
    const delay = Math.min(RECONNECT_BASE_DELAY_MS * 2 ** (attempt - 1), RECONNECT_MAX_DELAY_MS);
    console.log(`unexpected disconnect, reconnecting in ${delay}ms (attempt ${attempt}/${RECONNECT_MAX_ATTEMPTS})`);
    setSocketStatus("reconnecting");
    clearReconnectTimeout();
    reconnectTimeoutRef.current = setTimeout(() => {
      reconnectTimeoutRef.current = null;
      connectRef.current();
    }, delay);
  }, [onDisconnectProp, onReconnectExhausted, clearReconnectTimeout]);

  const onMessageEvent = useCallback(
    (eventData: MessageEvent) => {
      lastMessageTime.current = Date.now();
      const dataArray = new Uint8Array(eventData.data);
      const message = decodeMessage(dataArray);
      if (message.type == "handshake") {
        console.log("Handshake received, let's rocknroll.");
        // A successful handshake proves the new connection is fully healthy end to end;
        // reset the backoff so a later, unrelated blip starts counting from zero again.
        reconnectAttemptsRef.current = 0;
        setSocketStatus("connected");
      }
      if (!onMessage) {
        return;
      }
      onMessage(message);
    },
    [onMessage, setSocketStatus],
  );

  const connect = useCallback(() => {
    // Close existing socket if any
    if (socketRef.current) {
      console.log("closing existing socket before creating new one");
      const stale = socketRef.current;
      socketRef.current = null;
      stale.close();
    }
    manualCloseRef.current = false;
    const ws = new WebSocket(uri);
    ws.binaryType = "arraybuffer";
    ws.addEventListener("open", onConnect);
    ws.addEventListener("close", onDisconnect);
    ws.addEventListener("message", onMessageEvent);

    socketRef.current = ws;
    lastMessageTime.current = Date.now();
    console.log("Socket created", ws);
  }, [uri, onConnect, onDisconnect, onMessageEvent]);

  useEffect(() => {
    connectRef.current = connect;
  }, [connect]);

  const start = useCallback(() => {
    reconnectAttemptsRef.current = 0;
    clearReconnectTimeout();
    connect();
  }, [connect, clearReconnectTimeout]);

  const stop = useCallback(() => {
      manualCloseRef.current = true;
      clearReconnectTimeout();
      reconnectAttemptsRef.current = 0;
      if (socketRef.current) {
        socketRef.current.close();
        socketRef.current = null;
      }
      setSocketStatus("disconnected");
  }, [clearReconnectTimeout]);

  useEffect(() => {
    if(socketStatus !== "connected") {
      return;
    }
    const intervalId = setInterval(() => {
      if (lastMessageTime.current && Date.now() - lastMessageTime.current > INACTIVITY_TIMEOUT_MS) {
        console.log("closing socket due to inactivity", socketRef.current);
        socketRef.current?.close();
      }
    }, INACTIVITY_CHECK_INTERVAL_MS);

    return () => {
      clearInterval(intervalId);
    };
  }, [socketStatus]);

  useEffect(() => {
    return () => {
      clearReconnectTimeout();
    };
  }, [clearReconnectTimeout]);

  return {
    socketStatus,
    socket: socketRef.current,
    sendMessage,
    start,
    stop,
    setSocketStatus,
  };
};
