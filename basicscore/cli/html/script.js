function createTokenSpan(token, probability) {
  let spanToken = document.createElement('span')
  spanToken.textContent = token
  let gradient = probability * 255
  spanToken.style.backgroundColor = `rgba(${255 - gradient},${0 + gradient},0, 0.5)`
  return spanToken;
}

function mapToken(tokenStr) {
  if (tokenStr !== '\n') {
    return tokenStr;
  } else {
    return '\u2B90';
  }
}

function splitTokenLines(tokens) {
    let lines = [];
    var line = [];
    tokens.forEach(function(token){
        if (token[0] == '\n') {
            line.push(['\u2B90', token[1]])
            lines.push(line);
            line = [];
        } else {
            line.push(token);
        }
    });
    lines.push(line);
    return lines;
}

function createFilePre(tokens) {
  let pre = document.createElement('pre');
  pre.classList.add("prettyprint");
  pre.classList.add("lang-js");

  lines = splitTokenLines(tokens);
    lines.forEach(function (line){
        let linediv = document.createElement('div');
        linediv.classList.add("preline");
        line.forEach(function (tokenProb){
            linediv.appendChild(createTokenSpan(tokenProb[0], tokenProb[1]));
        });
        pre.appendChild(linediv);
    })

  return pre;
}

function createFileSection(filename, score, tokenProbs) {
  let div = document.createElement('div');
  let title = document.createElement('h2');
  title.innerText = filename+ ": " + score;
  div.appendChild(title);

  div.appendChild(createFilePre(tokenProbs));
  return div;
}

const scoreContainer = document.querySelector("#filescores");

file_data.forEach(function (filedata) {
    let entry = createFileSection(filedata.filename, filedata.score, filedata.tokens);
    scoreContainer.appendChild(entry)
  }
)
