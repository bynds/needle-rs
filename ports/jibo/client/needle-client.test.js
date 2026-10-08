/* Tests for needle-client.js against a fake server; ES5, built-in modules only.
 *
 *   node needle-client.test.js                     # fake server only
 *   NEEDLE_SOCKET=/path/to/needle.sock node needle-client.test.js   # also one real request
 */
'use strict';

var assert = require('assert');
var fs = require('fs');
var net = require('net');
var os = require('os');
var path = require('path');
var needle = require('./needle-client');

var sockPath = path.join(os.tmpdir(), 'needle-client-test-' + process.pid + '.sock');
var replies = [];
var received = [];
var server = net.createServer({allowHalfOpen: true}, function (c) {
  var buf = '';
  c.on('data', function (d) { buf += d; });
  c.on('end', function () {
    received.push(JSON.parse(buf));
    var r = replies.shift();
    if (r === 'hang') return; /* never answer */
    c.end(typeof r === 'string' ? r : JSON.stringify(r) + '\n');
  });
});

var tests = [];
function test(name, fn) { tests.push({name: name, fn: fn}); }

function handlersLog(log) {
  return {
    start_timer: function (a, done) { log.push('start ' + a.seconds); done(null, 'ok'); },
    cancel_timer: function (a, done) { log.push('cancel'); done(); }
  };
}

test('a candidate runs its handlers in order', function (next) {
  var log = [];
  replies.push({status: 'candidate', calls: [{name: 'cancel_timer', arguments: {}},
    {name: 'start_timer', arguments: {seconds: 600}}]});
  var c = needle.connect(sockPath, {timeoutMs: 2000});
  c.run('cancel and start ten minutes', {tools: ['start_timer'], handlers: handlersLog(log)},
    function (err, out) {
      assert.ifError(err);
      assert.strictEqual(out.action, 'executed');
      assert.deepStrictEqual(log, ['cancel', 'start 600']);
      var req = received[received.length - 1];
      assert.strictEqual(req.query, 'cancel and start ten minutes');
      assert.deepStrictEqual(req.tools, ['start_timer']);
      assert.ok(/^js-\d+$/.test(req.request_id));
      next();
    });
});

test('nothing but a candidate ever executes', function (next) {
  var statuses = {
    needs_clarification: 'clarify', low_confidence: 'clarify', no_call: 'none',
    unsupported: 'none', invalid_output: 'none', incomplete: 'none', busy: 'retry_later',
    timeout: 'retry_later', truncated: 'error', invalid_request: 'error', brand_new: 'error'
  };
  var keys = Object.keys(statuses);
  var log = [];
  (function each(i) {
    if (i === keys.length) {
      assert.deepStrictEqual(log, []);
      return next();
    }
    /* Even with calls attached, a non-candidate must not run them. */
    replies.push({status: keys[i], calls: [{name: 'start_timer', arguments: {seconds: 1}}],
      rejected_calls: [{name: 'start_timer', arguments: {seconds: 1800}}]});
    needle.connect(sockPath).run('x', {handlers: handlersLog(log)}, function (err, out) {
      assert.ifError(err);
      assert.strictEqual(out.action, statuses[keys[i]], keys[i]);
      each(i + 1);
    });
  })(0);
});

test('a missing handler means nothing runs at all', function (next) {
  var log = [];
  replies.push({status: 'candidate', calls: [{name: 'start_timer', arguments: {seconds: 5}},
    {name: 'unlock_door', arguments: {}}]});
  needle.connect(sockPath).run('x', {handlers: handlersLog(log)}, function (err, out) {
    assert.ok(err && /unlock_door/.test(err.message));
    assert.strictEqual(out.action, 'error');
    assert.deepStrictEqual(log, []);
    next();
  });
});

test('a handler named like an Object builtin is not found', function (next) {
  replies.push({status: 'candidate', calls: [{name: 'toString', arguments: {}}]});
  needle.connect(sockPath).run('x', {handlers: {}}, function (err, out) {
    assert.ok(err);
    assert.strictEqual(out.action, 'error');
    next();
  });
});

test('a handler error stops the sequence and says how far it got', function (next) {
  var log = [];
  replies.push({status: 'candidate', calls: [{name: 'a', arguments: {}}, {name: 'b', arguments: {}}]});
  needle.connect(sockPath).run('x', {handlers: {
    a: function (args, done) { done(new Error('motor busy')); },
    b: function (args, done) { log.push('b'); done(); }
  }}, function (err, out) {
    assert.ok(err && /motor busy/.test(err.message));
    assert.strictEqual(out.executed, 0);
    assert.deepStrictEqual(log, []);
    next();
  });
});

test('garbage and silence are errors, not actions', function (next) {
  replies.push('not json\n');
  needle.connect(sockPath).propose('x', {}, function (err) {
    assert.ok(err && /unreadable/.test(err.message));
    replies.push({no: 'status'});
    needle.connect(sockPath).propose('x', {}, function (err2) {
      assert.ok(err2 && /status/.test(err2.message));
      replies.push('hang');
      needle.connect(sockPath, {timeoutMs: 200}).run('x', {}, function (err3, out) {
        assert.ok(err3 && /did not answer/.test(err3.message));
        assert.strictEqual(out.action, 'retry_later');
        next();
      });
    });
  });
});

function realServer(next) {
  var real = process.env.NEEDLE_SOCKET;
  if (!real) return next();
  var c = needle.connect(real, {timeoutMs: 120000});
  c.health(function (err, h) {
    assert.ifError(err);
    assert.strictEqual(h.status, 'ok');
    c.run('Start a 90 second timer.', {tools: ['start_timer'], handlers: {
      start_timer: function (a, done) { done(null, a.seconds); }
    }}, function (err2, out) {
      assert.ifError(err2);
      console.log('real server: ' + out.action + ' ' + JSON.stringify(out.response.calls || out.response.detail));
      assert.strictEqual(out.action, 'executed');
      assert.deepStrictEqual(out.results, [90]);
      next();
    });
  });
}

try { fs.unlinkSync(sockPath); } catch (e) { /* none */ }
server.listen(sockPath, function () {
  (function run(i) {
    if (i === tests.length) {
      server.close();
      return realServer(function () { console.log('ok: ' + tests.length + ' tests' +
        (process.env.NEEDLE_SOCKET ? ' + real server' : '')); });
    }
    tests[i].fn(function () { console.log('ok   ' + tests[i].name); run(i + 1); });
  })(0);
});
