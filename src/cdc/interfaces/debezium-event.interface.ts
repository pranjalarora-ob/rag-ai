export interface DebeziumPayload<T> {
  before: T | null;
  after: T | null;
  op: 'c' | 'u' | 'd' | 'r';
}

export interface DebeziumEvent<T> {
  payload: DebeziumPayload<T>;
}