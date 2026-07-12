// The `zone` field in the source data is a regional-hub label that mixes true
// regions (North/South/East/West) with hub-city names (Gurgaon, Bangalore, Mumbai,
// Kolkata). Grouping by raw zone therefore splits one region across several buckets.
// toRegion() collapses the observed hub values into 4 canonical regions so zone-wise
// aggregates are clean. Records with no hub stay "Unassigned" (not force-mapped).
const ZONE_TO_REGION: Record<string, string> = {
  north: 'North',
  gurgaon: 'North',
  gurugram: 'North',
  south: 'South',
  bangalore: 'South',
  bengaluru: 'South',
  west: 'West',
  mumbai: 'West',
  east: 'East',
  kolkata: 'East',
};

export function toRegion(zone: any): string {
  const z = String(zone ?? '').trim().toLowerCase();
  if (!z || z === 'unassigned' || z === 'none' || z === 'null' || z === 'undefined') {
    return 'Unassigned';
  }
  return ZONE_TO_REGION[z] ?? String(zone).trim();
}
