/**
 * A neutral, human-readable title for a case.
 *
 * Cases used to be titled "Case of <confirmed finding>", which showed the
 * student the answer before they had written an interpretation. The title is
 * now built from the case id alone, so it says which case this is and
 * nothing about what is on the film.
 */
export function caseTitle(caseId: string): string {
  return `Case ${caseId.replace(/^case_/, "")}`;
}
