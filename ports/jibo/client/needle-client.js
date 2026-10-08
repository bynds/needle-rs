/* A client for `needle-jibo serve --socket`, for skills running on the robot's Node.js.
 *
 * Plain ES5 with callbacks and only the built-in `net` module, because the robot's Node version
 * is old and unknown here (the same constraint as the Decider port's decider-client.js, whose
 * shape this follows). One request per connection: write the JSON, close our side, read one
 * JSON line back.
 *
 *   var needle = require('./needle-client');
 *   var client = needle.connect('/path/to/needle.sock', {timeoutMs: 20000});
 *   client.run('set a timer for ten minutes', {
 *     tools: ['start_timer', 'cancel_timer'],        // a subset of the server's catalogue
 *     handlers: {                                     // the application's own functions
 *       start_timer: function (args, done) { timers.start(args.seconds * 1000); done(); },
 *       cancel_timer: function (args, done) { timers.cancel(); done(); }
 *     }
 *   }, function (err, outcome) {
 *     if (err) return log(err.message);
 *     // outcome.action: 'executed' | 'clarify' | 'none' | 'retry_later' | 'error'
 *     if (outcome.action === 'clarify') say('Sorry, how long should the timer be?');
 *   });
 *
 * Safety rules this module keeps:
 * - Only a response with status "candidate" ever reaches a handler. Everything else (an
 *   abstention, an ungrounded or low-confidence proposal, a malformed or unfinished answer, a
 *   refusal) is mapped to an action and never executed.
 * - Every call of a candidate must have a handler before any handler runs: all or nothing.
 * - Handlers receive the validated `arguments` object and nothing else; nothing generated is
 *   ever evaluated or passed to a shell.
 * - A timeout only stops waiting. The server finishes what it is computing, so do not resend
 *   at once; `retry_later` says so.
 */
'use strict';

var net = require('net');

var nextId = 1;

function request(socketPath, payload, timeoutMs, callback) {
  var done = false;
  var chunks = [];
  var size = 0;
  var sock = net.connect(socketPath);

  function finish(err, value) {
    if (done) return;
    done = true;
    clearTimeout(timer);
    sock.destroy();
    callback(err, value);
  }

  var timer = setTimeout(function () {
    finish(new Error('needle-jibo did not answer within ' + timeoutMs + ' ms'));
  }, timeoutMs);

  sock.on('error', function (err) { finish(err); });
  sock.on('data', function (chunk) {
    size += chunk.length;
    if (size > 65536) return finish(new Error('needle-jibo response too large'));
    chunks.push(chunk);
  });
  sock.on('end', function () {
    var text = Buffer.concat(chunks).toString('utf8');
    var body;
    try {
      body = JSON.parse(text);
    } catch (e) {
      return finish(new Error('unreadable response from needle-jibo: ' + text.slice(0, 200)));
    }
    if (!body || typeof body.status !== 'string') {
      return finish(new Error('response without a status: ' + text.slice(0, 200)));
    }
    finish(null, body);
  });
  sock.end(JSON.stringify(payload));
}

/* What the application should do with a response, without running anything. */
function classify(response) {
  switch (response.status) {
    case 'candidate':
      return 'execute';
    case 'needs_clarification':
    case 'low_confidence':
      return 'clarify';
    case 'no_call':
    case 'unsupported':
    case 'invalid_output':
    case 'incomplete':
      return 'none';
    case 'busy':
    case 'timeout':
      return 'retry_later';
    default: /* truncated, invalid_request, anything new */
      return 'error';
  }
}

/* Run the handlers of a candidate in order, all or nothing. callback(err, outcome). */
function dispatch(response, handlers, callback) {
  var action = classify(response);
  var outcome = {action: action, response: response};
  if (action !== 'execute') return callback(null, outcome);
  var calls = response.calls;
  if (!calls || !calls.length) {
    outcome.action = 'error';
    return callback(new Error('candidate without calls'), outcome);
  }
  var i;
  for (i = 0; i < calls.length; i++) {
    if (!handlers || !Object.prototype.hasOwnProperty.call(handlers, calls[i].name) ||
        typeof handlers[calls[i].name] !== 'function') {
      outcome.action = 'error';
      return callback(new Error('no handler for ' + calls[i].name + '; nothing executed'), outcome);
    }
  }
  var results = [];
  (function step(k) {
    if (k === calls.length) {
      outcome.action = 'executed';
      outcome.results = results;
      return callback(null, outcome);
    }
    var once = false;
    try {
      handlers[calls[k].name](calls[k].arguments || {}, function (err, result) {
        if (once) return;
        once = true;
        if (err) {
          outcome.action = 'error';
          outcome.executed = k;
          return callback(err, outcome);
        }
        results.push(result);
        step(k + 1);
      });
    } catch (e) {
      outcome.action = 'error';
      outcome.executed = k;
      callback(e, outcome);
    }
  })(0);
}

function connect(socketPath, options) {
  var timeoutMs = (options && options.timeoutMs) || 30000;
  function propose(query, opts, callback) {
    var payload = {request_id: (opts && opts.requestId) || ('js-' + (nextId++)), query: query};
    if (opts && opts.tools) payload.tools = opts.tools;
    if (opts && opts.maxNewTokens) payload.max_new_tokens = opts.maxNewTokens;
    request(socketPath, payload, timeoutMs, callback);
  }
  return {
    /* The raw response: no handler runs. */
    propose: propose,
    /* propose, then dispatch. */
    run: function (query, opts, callback) {
      propose(query, opts, function (err, response) {
        if (err) return callback(err, {action: 'retry_later', error: err.message});
        dispatch(response, opts && opts.handlers, callback);
      });
    },
    health: function (callback) { request(socketPath, {health: true}, timeoutMs, callback); }
  };
}

module.exports = {connect: connect, classify: classify, dispatch: dispatch};
