import jsonfile from "jsonfile";
import moment from "moment";
import simpleGit from "simple-git";
import random from "random";
import fs from "fs";

const files = [
  "data.json", "index.js", "README.md", "notes.txt", "config.json",
  "__init__.py", "environment.py", "high_level_actions.py", "LLM.py",
  "low_level_actions.py", "magent_runner.py", "retrieval.py", "runner.py",
  "schema.py", "test_model.py", "failure-case.md", "modular_ds_agent.py",
  "requirements.txt", "code_evaluation.sh", "code_generation.sh",
  "evaluation.py", "execution.py", "generate.py", "prompt.py",
  "start_api.sh", "housing.xlsx", "setup.py"
];

const getRandomFile = () => files[random.int(0, files.length - 1)];

const commitChange = (file, date, callback) => {
  // Create a new simple-git instance with env for dates
  const git = simpleGit().env({
    GIT_COMMITTER_DATE: date,
    GIT_AUTHOR_DATE: date,
  });

  const commitMessage = `Update ${file} on ${date}`;

  git
    .add(file)
    .commit(commitMessage, {
      "--date": date,
      "--author": `"SaeedSubayyal <subayyalsaeed321@gmail.com>"`,
    }, (err) => {
      if (err) {
        console.error("Commit error:", err);
        return;
      }
      callback();
    });
};

const makeCommits = (n) => {
  if (n === 0) {
    return simpleGit().push("origin", "main");
  }

  const startDate = moment("2025-01-01");
  const endDate = moment("2025-05-14");

  const daysDiff = endDate.diff(startDate, "days");
  const randomDays = random.int(0, daysDiff);
  const date = startDate.clone().add(randomDays, "days");

  if (date.isAfter(moment())) return makeCommits(n); // prevent future dates

  const formattedDate = date.format();
  const fileToModify = getRandomFile();
  const content = `// Updated by script on ${formattedDate}\n`;

  console.log(`Committing on ${formattedDate} to file ${fileToModify}`);

  if (fileToModify.endsWith(".json")) {
    jsonfile.writeFile(fileToModify, { updated_on: formattedDate }, { spaces: 2 }, (err) => {
      if (err) {
        console.error(err);
        return;
      }
      commitChange(fileToModify, formattedDate, () => makeCommits(n - 1));
    });
  } else {
    fs.appendFile(fileToModify, content, (err) => {
      if (err) {
        console.error(err);
        return;
      }
      commitChange(fileToModify, formattedDate, () => makeCommits(n - 1));
    });
  }
};

// Start creating commits
makeCommits(100);
// Updated by script on 2025-04-22T00:00:00+05:00
// Updated by script on 2025-01-20T00:00:00+05:00
// Updated by script on 2025-03-12T00:00:00+05:00
