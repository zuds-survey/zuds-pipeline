#!/bin/bash
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
CREATE USER decam_admin;
CREATE DATABASE decam;
GRANT ALL PRIVILEGES ON DATABASE decam TO decam_admin;
ALTER ROLE DECAM_ADMIN WITH SUPERUSER;
EOSQL

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "decam" <<-EOSQL
CREATE EXTENSION q3c;
EOSQL

