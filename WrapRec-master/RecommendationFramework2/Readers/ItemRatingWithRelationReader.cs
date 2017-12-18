﻿using RF2.Data;
using RF2.Entities;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RF2.Readers
{
    public class ItemRatingWithRelationReader : DatasetReaderWithFilter<ItemRatingWithRelations>
    {
        Dictionary<string, List<string>> _relations;
        IDatasetReader<ItemRating> _itemRatingsReader;

        public ItemRatingWithRelationReader(IDatasetReader<ItemRating> itemRatingReader, IEnumerable<Relation> relations)
        {
            _relations = relations.GroupBy(r => r.UserId).ToDictionary(g => g.Key, g => g.Select(r => r.ConnectedId).ToList());
            _itemRatingsReader = itemRatingReader;
        }
        
        public override IEnumerable<ItemRatingWithRelations> ReadWithoutFiltering()
        {
            return _itemRatingsReader.ReadSamples().Select(ir => new ItemRatingWithRelations(ir, _relations)).ToList();
        }
    }
}
