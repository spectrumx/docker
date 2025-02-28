FROM node:23
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npx update-browserslist-db@latest
EXPOSE 3000
CMD ["npm","start"]