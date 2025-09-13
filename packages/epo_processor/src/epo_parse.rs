use anyhow::{Context, Result};
use csv::WriterBuilder;
use indicatif::{ProgressBar, ProgressStyle};
use libxml::parser::Parser;
use libxml::tree::Node;
use libxml::xpath::Context as XpathContext;
use log::info;
use rayon::prelude::*;
use std::collections::BTreeSet;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use walkdir::WalkDir;

/// Represents a single patent exchange document with its metadata and classifications
#[derive(Debug)]
struct ExchangeDocument {
    country: String,
    doc_number: String,
    kind: String,
    status: String,
    patent_classifications: Option<PatentClassifications>,
    references_cited: Option<ReferencesCited>,
    patent_family: Option<PatentFamily>,
}

/// Container for patent classification information
#[derive(Debug)]
struct PatentClassifications {
    patent_classifications: Vec<PatentClassification>,
}

/// Represents a single patent classification entry
#[derive(Debug)]
struct PatentClassification {
    scheme: String,
    classification_symbol: String,
}

/// Container for cited references
#[derive(Debug)]
struct ReferencesCited {
    citations: Vec<Citation>,
}

/// Represents a single citation to another patent document
#[derive(Debug)]
struct Citation {
    cited_id: Option<String>,
    categories: Vec<String>,
}

/// Container for patent family information
#[derive(Debug)]
struct PatentFamily {
    family_members: Vec<FamilyMember>,
}

/// Represents a single family member in a patent family
#[derive(Debug)]
struct FamilyMember {
    publication_references: Vec<PublicationReference>,
}

/// Represents a publication reference in a patent family member
#[derive(Debug)]
struct PublicationReference {
    data_format: String,
    document_id: DocumentId,
}

/// Document identifier components for publication references
#[derive(Debug)]
struct DocumentId {
    country: String,
    doc_number: String,
    kind: String,
}

/// Parses all XML files in a directory and writes extracted data to a CSV file
///
/// # Arguments
/// * `download_dir` - Path to directory containing XML files
/// * `output_csv` - Path where output CSV will be created
/// * `batch_size`` - no. of files to process in parallel
///
/// # Returns
/// * `Result<()>` - Ok on success, Err on failure
pub fn parse_all_to_csv(download_dir: &str, output_csv: &str, batch_size: usize) -> Result<()> {
    let progress_bar = ProgressBar::new(0);
    let xml_files: Vec<_> = WalkDir::new(download_dir)
        .follow_links(true)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("xml"))
        .collect();

    progress_bar.set_length(xml_files.len() as u64);
    progress_bar.set_style(
        ProgressStyle::default_bar()
            .template("{msg} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap()
            .progress_chars("#>-"),
    );
    progress_bar.set_message("Parsing XML files");

    let output_file = File::create(output_csv).context("Failed to create output CSV file")?;
    let mut writer = WriterBuilder::new()
        .has_headers(true)
        .quote_style(csv::QuoteStyle::Always)
        .from_writer(BufWriter::new(output_file));

    writer
        .write_record(&[
            "patent_id",
            "status",
            "cpc_list",
            "citations",
            "family_patents",
        ])
        .context("Failed to write CSV header")?;

    for chunk in xml_files.chunks(batch_size) {
        let chunk_records: Result<Vec<Vec<Vec<String>>>> = chunk
            .par_iter()
            .map(|entry| process_single_xml(entry.path()))
            .collect();

        let records = chunk_records?;
        for record_batch in records {
            for record in record_batch {
                writer
                    .write_record(&record)
                    .context("Failed to write CSV record")?;
            }
        }

        progress_bar.inc(chunk.len() as u64);
    }

    progress_bar.finish_with_message("Done parsing");
    writer.flush().context("Failed to flush writer")?;
    Ok(())
}

/// Processes a single XML file and extracts patent data
///
/// This function parses an XML file and processes all exchange-document nodes
/// Each node is processed independently to extract
/// patent information including classifications, citations, and family patents.
///
/// # Arguments
/// * `xml_path` - Path to the XML file to process
///
/// # Returns
/// * `Result<Vec<Vec<String>>>` - Vector of records with extracted patent data
fn process_single_xml(xml_path: &Path) -> Result<Vec<Vec<String>>> {
    info!("Parsing XML: {}", xml_path.display());
    let parser = Parser::default();
    let doc = parser.parse_file(xml_path.to_str().context("Invalid path")?)?;
    let mut ctx =
        XpathContext::new(&doc).map_err(|_| anyhow::anyhow!("Failed to create XPath context"))?;

    let exchange_nodes = ctx
        .findnodes("//*[local-name()='exchange-document']", None)
        .map_err(|_| anyhow::anyhow!("Failed to find exchange documents"))?;

    let records: Vec<Vec<String>> = exchange_nodes
        .into_iter()
        .map(|node| {
            let exchange_doc = ExchangeDocument::from_node(&node, &mut ctx)
                .context("Failed to parse exchange document")?;

            let patent_id = format!(
                "{}{}{}",
                exchange_doc.country, exchange_doc.doc_number, exchange_doc.kind
            );

            let cpc_list = exchange_doc
                .patent_classifications
                .map_or(String::new(), |pc| {
                    pc.patent_classifications
                        .into_iter()
                        .filter(|p| p.scheme == "CPCI")
                        .map(|p| p.classification_symbol.trim().to_string())
                        .collect::<Vec<_>>()
                        .join(";")
                });

            let citations = exchange_doc.references_cited.map_or(String::new(), |rc| {
                rc.citations
                    .into_iter()
                    .filter_map(|c| {
                        c.cited_id.as_ref().map(|cid| {
                            let cats = c.categories.join(",");
                            format!("{} ({})", cid, cats)
                        })
                    })
                    .collect::<Vec<_>>()
                    .join(";")
            });

            let family_patents = exchange_doc.patent_family.map_or(String::new(), |pf| {
                pf.family_members
                    .into_iter()
                    .flat_map(|fm| {
                        fm.publication_references
                            .into_iter()
                            .filter(|pr| pr.data_format == "docdb")
                            .map(|pr| {
                                format!(
                                    "{}{}{}",
                                    pr.document_id.country,
                                    pr.document_id.doc_number,
                                    pr.document_id.kind
                                )
                            })
                    })
                    .filter(|fid| *fid != patent_id)
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect::<Vec<_>>()
                    .join(";")
            });

            Ok(vec![
                patent_id,
                exchange_doc.status,
                cpc_list,
                citations,
                family_patents,
            ])
        })
        .collect::<Result<_>>()?;

    Ok(records)
}

impl ExchangeDocument {
    /// Creates an ExchangeDocument from an XML node using XPath context
    fn from_node(node: &Node, ctx: &mut XpathContext) -> Result<Self> {
        Ok(Self {
            country: node
                .get_attribute("country")
                .context("Missing country attribute")?,
            doc_number: node
                .get_attribute("doc-number")
                .context("Missing doc-number attribute")?,
            kind: node
                .get_attribute("kind")
                .context("Missing kind attribute")?,
            status: node
                .get_attribute("status")
                .context("Missing status attribute")?,
            patent_classifications: PatentClassifications::from_node(node, ctx).ok(),
            references_cited: ReferencesCited::from_node(node, ctx).ok(),
            patent_family: PatentFamily::from_node(node, ctx).ok(),
        })
    }
}

impl PatentClassifications {
    /// Extracts patent classifications from an XML node
    fn from_node(parent: &Node, ctx: &mut XpathContext) -> Result<Self> {
        let nodes = ctx
            .findnodes(".//*[local-name()='patent-classifications']/*[local-name()='patent-classification']", Some(parent))
            .map_err(|_| anyhow::anyhow!("Failed to find patent classifications"))?;
        let classifications = nodes
            .iter()
            .map(|n| PatentClassification::from_node(n, ctx))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            patent_classifications: classifications,
        })
    }
}

impl PatentClassification {
    /// Creates a PatentClassification from an XML node
    fn from_node(node: &Node, ctx: &mut XpathContext) -> Result<Self> {
        let scheme_node = ctx
            .findnodes("*[local-name()='classification-scheme']", Some(node))
            .map_err(|_| anyhow::anyhow!("Failed to find classification scheme"))?
            .into_iter()
            .next()
            .context("No classification scheme found")?;
        let scheme = scheme_node
            .get_attribute("scheme")
            .context("Missing scheme attribute")?;
        let symbol_node = ctx
            .findnodes("*[local-name()='classification-symbol']", Some(node))
            .map_err(|_| anyhow::anyhow!("Failed to find classification symbol"))?
            .into_iter()
            .next()
            .context("No classification symbol found")?;
        Ok(Self {
            scheme,
            classification_symbol: symbol_node.get_content(),
        })
    }
}

impl ReferencesCited {
    /// Extracts cited references from an XML node
    fn from_node(parent: &Node, ctx: &mut XpathContext) -> Result<Self> {
        let nodes = ctx
            .findnodes(
                ".//*[local-name()='references-cited']/*[local-name()='citation']",
                Some(parent),
            )
            .map_err(|_| anyhow::anyhow!("Failed to find citations"))?;
        let citations = nodes
            .iter()
            .map(|n| Citation::from_node(n, ctx))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { citations })
    }
}

impl Citation {
    /// Creates a Citation from an XML node
    fn from_node(node: &Node, ctx: &mut XpathContext) -> Result<Self> {
        let categories = extract_categories(node, ctx);

        let cited_id = ctx
            .findnodes("*[local-name()='patcit']", Some(node))
            .ok()
            .and_then(|patcit_nodes| patcit_nodes.into_iter().next())
            .and_then(|patcit| {
                ctx.findnodes("*[local-name()='document-id']", Some(&patcit))
                    .ok()
                    .and_then(|doc_id_nodes| doc_id_nodes.into_iter().next())
            })
            .and_then(|doc_id| {
                let country = extract_text_content(&doc_id, ctx, "*[local-name()='country']");
                let doc_number = extract_text_content(&doc_id, ctx, "*[local-name()='doc-number']");
                let kind = extract_text_content(&doc_id, ctx, "*[local-name()='kind']");

                if country.is_empty() && doc_number.is_empty() && kind.is_empty() {
                    None
                } else {
                    Some(format!("{}{}{}", country, doc_number, kind))
                }
            });

        Ok(Self {
            cited_id,
            categories,
        })
    }
}

/// Extracts citation categories from an XML node
fn extract_categories(node: &Node, ctx: &mut XpathContext) -> Vec<String> {
    let direct_nodes = ctx
        .findnodes("*[local-name()='category']", Some(node))
        .unwrap_or_default();

    let direct_categories = direct_nodes
        .iter()
        .map(|n| n.get_content().trim().to_string())
        .filter(|s| !s.is_empty());

    let rel_passage_nodes = ctx
        .findnodes("*[local-name()='rel-passage']", Some(node))
        .unwrap_or_default();

    let rel_passage_categories = rel_passage_nodes.iter().flat_map(|rp_node| {
        let category_nodes = ctx
            .findnodes("*[local-name()='category']", Some(rp_node))
            .unwrap_or_default();

        category_nodes
            .iter()
            .map(|n| n.get_content().trim().to_string())
            .filter(|s| !s.is_empty())
            .collect::<Vec<String>>()
    });

    direct_categories.chain(rel_passage_categories).collect()
}

/// Extracts text content from a node using XPath
fn extract_text_content(node: &Node, ctx: &mut XpathContext, xpath: &str) -> String {
    ctx.findnodes(xpath, Some(node))
        .unwrap_or_default()
        .iter()
        .next()
        .map(|n| n.get_content().trim().to_string())
        .unwrap_or_default()
}

impl PatentFamily {
    /// Extracts patent family information from an XML node
    fn from_node(parent: &Node, ctx: &mut XpathContext) -> Result<Self> {
        let nodes = ctx
            .findnodes(
                ".//*[local-name()='patent-family']/*[local-name()='family-member']",
                Some(parent),
            )
            .map_err(|_| anyhow::anyhow!("Failed to find family members"))?;
        let members = nodes
            .iter()
            .map(|n| FamilyMember::from_node(n, ctx))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            family_members: members,
        })
    }
}

impl FamilyMember {
    /// Extracts family member information from an XML node
    fn from_node(parent: &Node, ctx: &mut XpathContext) -> Result<Self> {
        let nodes = ctx
            .findnodes("*[local-name()='publication-reference']", Some(parent))
            .map_err(|_| anyhow::anyhow!("Failed to find publication references"))?;
        let refs = nodes
            .iter()
            .map(|n| PublicationReference::from_node(n, ctx))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            publication_references: refs,
        })
    }
}

impl PublicationReference {
    /// Creates a PublicationReference from an XML node
    fn from_node(node: &Node, ctx: &mut XpathContext) -> Result<Self> {
        let data_format = node
            .get_attribute("data-format")
            .context("Missing data-format attribute")?
            .to_string();

        let doc_id_node = ctx
            .findnodes("*[local-name()='document-id']", Some(node))
            .map_err(|_| anyhow::anyhow!("Failed to find document-id"))?
            .into_iter()
            .next()
            .context("No document-id found")?;

        let country = ctx
            .findnodes("*[local-name()='country']", Some(&doc_id_node))
            .map_err(|_| anyhow::anyhow!("Failed to find country"))?
            .into_iter()
            .next()
            .map(|n| n.get_content())
            .unwrap_or_default();

        let doc_number = ctx
            .findnodes("*[local-name()='doc-number']", Some(&doc_id_node))
            .map_err(|_| anyhow::anyhow!("Failed to find doc-number"))?
            .into_iter()
            .next()
            .map(|n| n.get_content())
            .unwrap_or_default();

        let kind = ctx
            .findnodes("*[local-name()='kind']", Some(&doc_id_node))
            .map_err(|_| anyhow::anyhow!("Failed to find kind"))?
            .into_iter()
            .next()
            .map(|n| n.get_content())
            .unwrap_or_default();

        Ok(Self {
            data_format,
            document_id: DocumentId {
                country,
                doc_number,
                kind,
            },
        })
    }
}
