const fs = require('fs');

module.exports = async ({github, context}) => {
  const body = fs.readFileSync(process.env.COMMENT_PATH, 'utf8');
  const marker = body.split('\n', 1)[0];
  const issue_number = Number(process.env.PR_NUMBER);
  const comments = await github.paginate(github.rest.issues.listComments, {
    ...context.repo, issue_number, per_page: 100,
  });
  const existing = comments.find(c =>
    c.user.login === 'github-actions[bot]' && c.body.startsWith(marker));
  if (process.env.PR_HEAD) {
    const {data: pr} = await github.rest.pulls.get({...context.repo, pull_number: issue_number});
    if (pr.head.sha !== process.env.PR_HEAD) return;
  }
  if (existing) {
    await github.rest.issues.updateComment({
      ...context.repo, comment_id: existing.id, body,
    });
  } else {
    await github.rest.issues.createComment({
      ...context.repo, issue_number, body,
    });
  }
};
