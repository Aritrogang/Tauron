const fs = require('fs');
const file = '/Users/krishjana/.gemini/antigravity/brain/b0e16eb9-1dcf-47e9-8e00-7051eb41f1d9/task.md';
let c = fs.readFileSync(file, 'utf8');
c = c.replace('- [ ] Update `Layout.js`', '- [x] Update `Layout.js`');
c = c.replace('- [ ] Create a `LandingPage` UI component', '- [x] Create a `LandingPage` UI component');
c = c.replace('- [ ] Include the Tauron logo', '- [x] Include the Tauron logo');
c = c.replace('- [ ] When \'Get Started\' is clicked', '- [x] When \'Get Started\' is clicked');
fs.writeFileSync(file, c);
