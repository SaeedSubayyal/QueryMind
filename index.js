import jsonfile from "jsonfile";
import moment from "moment";
import simpleGit from "simple-git";
import random from "random";
import fs from "fs";

// List of real project files
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

const makeCommits = (n) => {
  if (n === 0) return simpleGit().push("origin", "main");

  const startDate = moment("2025-01-01");
  const endDate = moment("2025-05-14");

  const daysDiff = endDate.diff(startDate, "days");
  const randomDays = random.int(0, daysDiff);
  const date = startDate.clone().add(randomDays, "days");

  // Prevent future dates
  if (date.isAfter(moment())) return makeCommits(n);

  const formattedDate = date.format();
  const fileToModify = getRandomFile();
  const content = `// Updated by script on ${formattedDate}\n`;

  console.log(`Committing on ${formattedDate} to file ${fileToModify}`);

  // Append or write content depending on file type
  if (fileToModify.endsWith(".json")) {
    jsonfile.writeFile(fileToModify, { updated_on: formattedDate }, { spaces: 2 }, (err) => {
      if (err) return console.error(err);
      commitChange(fileToModify, formattedDate, () => makeCommits(n - 1));
    });
  } else {
    fs.appendFile(fileToModify, content, (err) => {
      if (err) return console.error(err);
      commitChange(fileToModify, formattedDate, () => makeCommits(n - 1));
    });
  }
};

const commitChange = (file, date, callback) => {
  const git = simpleGit();
  const commitMessage = `Update ${file} on ${date}`;
  git
    .add(file)
    .commit(commitMessage, {
      "--date": date,
      "--author": `"SaeedSubayyal <subayyalsaeed321@gmail.com>"`
    }, (err) => {
      if (err) return console.error("Commit error:", err);
      callback();
    });
};

// Start creating commits
makeCommits(100);
// Updated by script on 2025-02-16T00:00:00+05:00
// Updated by script on 2025-02-19T00:00:00+05:00
