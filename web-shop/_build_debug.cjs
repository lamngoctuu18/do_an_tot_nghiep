const vite = require('vite');
vite.build().then(() => {
  console.log('BUILD OK');
}).catch(e => {
  console.log('=== FAIL ===');
  try { console.log(JSON.stringify(e, Object.getOwnPropertyNames(e), 2)); } catch(x) { console.log('stringify err:', x.message); }
  console.log('--- raw:');
  console.log(e);
  process.exit(1);
});
