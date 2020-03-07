begin;
\timing

alter table forcedphotometry drop constraint "imagefk";
alter table forcedphotometry drop constraint "sourcefk";
alter table forcedphotometry drop constraint "usource_image";
drop index "uimage_source";

\copy forcedphotometry (source_id, image_id, flux, fluxerr, flags, ra, dec) from 'output.csv' with csv header;

alter table forcedphotometry add constraint "imagefk" FOREIGN KEY (image_id)
  REFERENCES singleepochsubtractions (id) ON DELETE CASCADE;

alter table forcedphotometry add constraint "sourcefk" FOREIGN KEY (source_id)
  REFERENCES sources (id) ON DELETE CASCADE;

create unique index "usource_image" on forcedphotometry (source_id, image_id);
create index "uimage_source" on forcedphotometry (image_id, source_id);

commit;
